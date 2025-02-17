# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import chromadb
from numpy.typing import NDArray

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.inline.vector_io.chroma import ChromaVectorIOConfig as InlineChromaVectorIOConfig
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import ChromaVectorIOConfig as RemoteChromaVectorIOConfig

log = logging.getLogger(__name__)


ChromaClientType = Union[chromadb.AsyncHttpClient, chromadb.PersistentClient]


# this is a helper to allow us to use async and non-async chroma clients interchangeably
async def maybe_await(result):
    if asyncio.iscoroutine(result):
        return await result
    return result


class ChromaIndex(EmbeddingIndex):
    def __init__(self, client: ChromaClientType, collection):
        self.client = client
        self.collection = collection

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        ids = [f"{c.metadata['document_id']}:chunk-{i}" for i, c in enumerate(chunks)]
        await maybe_await(
            self.collection.add(
                documents=[chunk.model_dump_json() for chunk in chunks],
                embeddings=embeddings,
                ids=ids,
            )
        )

    async def query(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        results = await maybe_await(
            self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=k,
                include=["documents", "distances"],
            )
        )
        distances = results["distances"][0]
        documents = results["documents"][0]

        chunks = []
        scores = []
        for dist, doc in zip(distances, documents, strict=False):
            try:
                doc = json.loads(doc)
                chunk = Chunk(**doc)
            except Exception:
                log.exception(f"Failed to parse document: {doc}")
                continue

            chunks.append(chunk)
            scores.append(1.0 / float(dist))

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        await maybe_await(self.client.delete_collection(self.collection.name))


class ChromaVectorIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self,
        config: Union[RemoteChromaVectorIOConfig, InlineChromaVectorIOConfig],
        inference_api: Api.inference,
    ) -> None:
        log.info(f"Initializing ChromaVectorIOAdapter with url: {config}")
        self.config = config
        self.inference_api = inference_api

        self.client = None
        self.cache = {}

    async def initialize(self) -> None:
        if isinstance(self.config, RemoteChromaVectorIOConfig):
            log.info(f"Connecting to Chroma server at: {self.config.url}")
            url = self.config.url.rstrip("/")
            parsed = urlparse(url)

            if parsed.path and parsed.path != "/":
                raise ValueError("URL should not contain a path")

            self.client = await chromadb.AsyncHttpClient(host=parsed.hostname, port=parsed.port)
        else:
            log.info(f"Connecting to Chroma local db at: {self.config.db_path}")
            self.client = chromadb.PersistentClient(path=self.config.db_path)

    async def shutdown(self) -> None:
        pass

    async def register_vector_db(
        self,
        vector_db: VectorDB,
    ) -> None:
        collection = await maybe_await(
            self.client.get_or_create_collection(
                name=vector_db.identifier,
                metadata={"vector_db": vector_db.model_dump_json()},
            )
        )
        self.cache[vector_db.identifier] = VectorDBWithIndex(
            vector_db, ChromaIndex(self.client, collection), self.inference_api
        )

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: List[Chunk],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)

        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)

        return await index.query_chunks(query, params)

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        if not vector_db:
            raise ValueError(f"Vector DB {vector_db_id} not found in Llama Stack")
        collection = await maybe_await(self.client.get_collection(vector_db_id))
        if not collection:
            raise ValueError(f"Vector DB {vector_db_id} not found in Chroma")
        index = VectorDBWithIndex(vector_db, ChromaIndex(self.client, collection), self.inference_api)
        self.cache[vector_db_id] = index
        return index
