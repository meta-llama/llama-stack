# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Union

from numpy.typing import NDArray
from pymilvus import MilvusClient

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.inline.vector_io.milvus import MilvusVectorIOConfig as InlineMilvusVectorIOConfig
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import MilvusVectorIOConfig as RemoteMilvusVectorIOConfig

logger = logging.getLogger(__name__)


class MilvusIndex(EmbeddingIndex):
    def __init__(self, client: MilvusClient, collection_name: str, consistency_level="Strong"):
        self.client = client
        self.collection_name = collection_name.replace("-", "_")
        self.consistency_level = consistency_level

    async def delete(self):
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                self.collection_name,
                dimension=len(embeddings[0]),
                auto_id=True,
                consistency_level=self.consistency_level,
            )

        data = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk_id = generate_chunk_id(chunk.metadata["document_id"], chunk.content)

            data.append(
                {
                    "chunk_id": chunk_id,
                    "vector": embedding,
                    "chunk_content": chunk.model_dump(),
                }
            )
        try:
            self.client.insert(
                self.collection_name,
                data=data,
            )
        except Exception as e:
            logger.error(f"Error inserting chunks into Milvus collection {self.collection_name}: {e}")
            raise e

    async def query(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        search_res = self.client.search(
            collection_name=self.collection_name,
            data=[embedding],
            limit=k,
            output_fields=["*"],
            search_params={"params": {"radius": score_threshold}},
        )
        chunks = [Chunk(**res["entity"]["chunk_content"]) for res in search_res[0]]
        scores = [res["distance"] for res in search_res[0]]
        return QueryChunksResponse(chunks=chunks, scores=scores)


class MilvusVectorIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self, config: Union[RemoteMilvusVectorIOConfig, InlineMilvusVectorIOConfig], inference_api: Api.inference
    ) -> None:
        self.config = config
        self.cache = {}
        self.client = None
        self.inference_api = inference_api

    async def initialize(self) -> None:
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            logger.info(f"Connecting to Milvus server at {self.config.uri}")
            self.client = MilvusClient(**self.config.model_dump(exclude_none=True))
        else:
            logger.info(f"Connecting to Milvus Lite at: {self.config.db_path}")
            uri = os.path.expanduser(self.config.db_path)
            self.client = MilvusClient(uri=uri)

    async def shutdown(self) -> None:
        self.client.close()

    async def register_vector_db(
        self,
        vector_db: VectorDB,
    ) -> None:
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            consistency_level = self.config.consistency_level
        else:
            consistency_level = "Strong"
        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=MilvusIndex(self.client, vector_db.identifier, consistency_level=consistency_level),
            inference_api=self.inference_api,
        )

        self.cache[vector_db.identifier] = index

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> Optional[VectorDBWithIndex]:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        if not vector_db:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=MilvusIndex(client=self.client, collection_name=vector_db.identifier),
            inference_api=self.inference_api,
        )
        self.cache[vector_db_id] = index
        return index

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        if vector_db_id in self.cache:
            await self.cache[vector_db_id].index.delete()
            del self.cache[vector_db_id]

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: List[Chunk],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        return await index.query_chunks(query, params)


def generate_chunk_id(document_id: str, chunk_text: str) -> str:
    """Generate a unique chunk ID using a hash of document ID and chunk text."""
    hash_input = f"{document_id}:{chunk_text}".encode("utf-8")
    return str(uuid.UUID(hashlib.md5(hash_input).hexdigest()))


# TODO: refactor this generate_chunk_id along with the `sqlite-vec` implementation into a separate utils file
