# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import heapq
import json
from typing import Any
from urllib.parse import urlparse

import chromadb
from numpy.typing import NDArray

from llama_stack.apis.files import Files
from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.inline.vector_io.chroma import ChromaVectorIOConfig as InlineChromaVectorIOConfig
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    ChunkForDeletion,
    EmbeddingIndex,
    VectorDBWithIndex,
)
from llama_stack.providers.utils.vector_io.vector_utils import Reranker

from .config import ChromaVectorIOConfig as RemoteChromaVectorIOConfig

log = get_logger(name=__name__, category="vector_io::chroma")

ChromaClientType = chromadb.api.AsyncClientAPI | chromadb.api.ClientAPI

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_dbs:chroma:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:chroma:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:chroma:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:chroma:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:chroma:{VERSION}::"


# this is a helper to allow us to use async and non-async chroma clients interchangeably
async def maybe_await(result):
    if asyncio.iscoroutine(result):
        return await result
    return result


class ChromaIndex(EmbeddingIndex):
    def __init__(self, client: ChromaClientType, collection, kvstore: KVStore | None = None):
        self.client = client
        self.collection = collection
        self.kvstore = kvstore

    async def initialize(self):
        pass

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        ids = [f"{c.metadata.get('document_id', '')}:{c.chunk_id}" for c in chunks]
        await maybe_await(
            self.collection.add(
                documents=[chunk.model_dump_json() for chunk in chunks],
                embeddings=embeddings,
                ids=ids,
            )
        )

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
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

            score = 1.0 / float(dist) if dist != 0 else float("inf")
            if score < score_threshold:
                continue

            chunks.append(chunk)
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        await maybe_await(self.client.delete_collection(self.collection.name))

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        results = await maybe_await(
            self.collection.query(
                query_texts=[query_string],
                where_document={"$contains": query_string},
                n_results=k,
                include=["documents", "distances"],
            )
        )

        distances = results["distances"][0] if results["distances"] else []
        documents = results["documents"][0] if results["documents"] else []

        chunks = []
        scores = []

        for dist, doc in zip(distances, documents, strict=False):
            try:
                doc_data = json.loads(doc)
                chunk = Chunk(**doc_data)
            except Exception:
                log.exception(f"Failed to parse document: {doc}")
                continue

            score = 1.0 / (1.0 + float(dist)) if dist is not None else 1.0

            if score < score_threshold:
                continue

            chunks.append(chunk)
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete a single chunk from the Chroma collection by its ID."""
        ids = [f"{chunk.document_id}:{chunk.chunk_id}" for chunk in chunks_for_deletion]
        await maybe_await(self.collection.delete(ids=ids))

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        """
        Hybrid search combining vector similarity and keyword search using configurable reranking.
        Args:
            embedding: The query embedding vector
            query_string: The text query for keyword search
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            reranker_type: Type of reranker to use ("rrf" or "weighted")
            reranker_params: Parameters for the reranker
        Returns:
            QueryChunksResponse with combined results
        """
        if reranker_params is None:
            reranker_params = {}

        # Get results from both search methods
        vector_response = await self.query_vector(embedding, k, score_threshold)
        keyword_response = await self.query_keyword(query_string, k, score_threshold)

        # Convert responses to score dictionaries using chunk_id
        vector_scores = {
            chunk.chunk_id: score for chunk, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            chunk.chunk_id: score
            for chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        # Combine scores using the reranking utility
        combined_scores = Reranker.combine_search_results(vector_scores, keyword_scores, reranker_type, reranker_params)

        # Efficient top-k selection because it only tracks the k best candidates it's seen so far
        top_k_items = heapq.nlargest(k, combined_scores.items(), key=lambda x: x[1])

        # Filter by score threshold
        filtered_items = [(doc_id, score) for doc_id, score in top_k_items if score >= score_threshold]

        # Create a map of chunk_id to chunk for both responses
        chunk_map = {c.chunk_id: c for c in vector_response.chunks + keyword_response.chunks}

        # Use the map to look up chunks by their IDs
        chunks = []
        scores = []
        for doc_id, score in filtered_items:
            if doc_id in chunk_map:
                chunks.append(chunk_map[doc_id])
                scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)


class ChromaVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self,
        config: RemoteChromaVectorIOConfig | InlineChromaVectorIOConfig,
        inference_api: Api.inference,
        files_api: Files | None,
    ) -> None:
        log.info(f"Initializing ChromaVectorIOAdapter with url: {config}")
        self.config = config
        self.inference_api = inference_api
        self.client = None
        self.cache = {}
        self.kvstore: KVStore | None = None
        self.vector_db_store = None
        self.files_api = files_api

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        self.vector_db_store = self.kvstore

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
        self.openai_vector_stores = await self._load_openai_vector_stores()

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
        if vector_db_id not in self.cache:
            log.warning(f"Vector DB {vector_db_id} not found")
            return

        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if index is None:
            raise ValueError(f"Vector DB {vector_db_id} not found in Chroma")

        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)

        if index is None:
            raise ValueError(f"Vector DB {vector_db_id} not found in Chroma")

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

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from a Chroma vector store."""
        index = await self._get_and_cache_vector_db_index(store_id)
        if not index:
            raise ValueError(f"Vector DB {store_id} not found")

        await index.index.delete_chunks(chunks_for_deletion)
