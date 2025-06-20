# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import logging
import os
from typing import Any

from numpy.typing import NDArray
from pymilvus import DataType, MilvusClient

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorIO,
)
from llama_stack.apis.vector_io.vector_io import (
    VectorStoreChunkingStrategy,
    VectorStoreFileContentsResponse,
    VectorStoreFileObject,
    VectorStoreListFilesResponse,
)
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.inline.vector_io.milvus import MilvusVectorIOConfig as InlineMilvusVectorIOConfig
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)
from llama_stack.providers.utils.vector_io.chunk_utils import generate_chunk_id

from .config import MilvusVectorIOConfig as RemoteMilvusVectorIOConfig

logger = logging.getLogger(__name__)


class MilvusIndex(EmbeddingIndex):
    def __init__(self, client: MilvusClient, collection_name: str, consistency_level="Strong"):
        self.client = client
        self.collection_name = collection_name.replace("-", "_")
        self.consistency_level = consistency_level

    async def delete(self):
        if await asyncio.to_thread(self.client.has_collection, self.collection_name):
            await asyncio.to_thread(self.client.drop_collection, collection_name=self.collection_name)

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )
        if not await asyncio.to_thread(self.client.has_collection, self.collection_name):
            await asyncio.to_thread(
                self.client.create_collection,
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
            await asyncio.to_thread(
                self.client.insert,
                self.collection_name,
                data=data,
            )
        except Exception as e:
            logger.error(f"Error inserting chunks into Milvus collection {self.collection_name}: {e}")
            raise e

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        search_res = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[embedding],
            limit=k,
            output_fields=["*"],
            search_params={"params": {"radius": score_threshold}},
        )
        chunks = [Chunk(**res["entity"]["chunk_content"]) for res in search_res[0]]
        scores = [res["distance"] for res in search_res[0]]
        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Keyword search is not supported in Milvus")

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Hybrid search is not supported in Milvus")


class MilvusVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self, config: RemoteMilvusVectorIOConfig | InlineMilvusVectorIOConfig, inference_api: Api.inference
    ) -> None:
        self.config = config
        self.cache = {}
        self.client = None
        self.inference_api = inference_api
        self.vector_db_store = None
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}
        self.files_api = None  # Files API is not yet available for Milvus
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            logger.info(f"Connecting to Milvus server at {self.config.uri}")
            self.client = MilvusClient(**self.config.model_dump(exclude_none=True))
        else:
            logger.info(f"Connecting to Milvus Lite at: {self.config.db_path}")
            uri = os.path.expanduser(self.config.db_path)
            self.client = MilvusClient(uri=uri)

        self.openai_vector_stores = await self._load_openai_vector_stores()

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

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex | None:
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
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        return await index.query_chunks(query, params)

    async def _save_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        try:
            if not await asyncio.to_thread(self.client.has_collection, self.metadata_collection_name):
                metadata_schema = MilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                    description="Metadata for OpenAI vector stores",
                )
                metadata_schema.add_field(
                    field_name="store_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512
                )  # max length for Milvus primary key
                metadata_schema.add_field(
                    field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1
                )  # required by Milvus
                metadata_schema.add_field(
                    field_name="metadata", datatype=DataType.VARCHAR, max_length=65535
                )  # max possible length in Milvus

                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=self.metadata_collection_name,
                    schema=metadata_schema,
                )

            data = [{"store_id": store_id, "vector": [0], "metadata": json.dumps(store_info)}]
            await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.metadata_collection_name,
                data=data,
            )
            self.openai_vector_stores[store_id] = store_info

        except Exception as e:
            logger.error(f"Error saving openai vector store {store_id}: {e}")
            raise

    async def _load_openai_vector_stores(self) -> dict[str, dict[str, Any]]:
        """Load the OpenAI vector store Milvus metadata collection."""
        openai_vector_stores = {}
        try:
            has_collection = await asyncio.to_thread(self.client.has_collection, self.metadata_collection_name)
            if not has_collection:
                return openai_vector_stores

            metadata_stats = await asyncio.to_thread(
                self.client.get_collection_stats,
                self.metadata_collection_name,
            )
            metadata_row_count = metadata_stats.get("row_count", 0)

            if metadata_row_count == 0:
                return openai_vector_stores

            collection_iterator = await asyncio.to_thread(
                self.client.query_iterator,
                collection_name=self.metadata_collection_name,
                batch_size=100,
            )
            try:
                while True:
                    result = collection_iterator.next()
                    if not result:
                        break

                    for row in result:
                        store_id = row.get("store_id")
                        if store_id:
                            try:
                                store_info = json.loads(row.get("metadata", "{}"))
                                openai_vector_stores[store_id] = store_info
                            except json.JSONDecodeError:
                                logger.error(f"failed to decode metadata for store_id {store_id}")
            finally:
                collection_iterator.close()
        except Exception as e:
            logger.error(f"error loading openai vector stores: {e}")

        return openai_vector_stores

    async def _update_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Update the OpenAI vector store Milvus metadata collection."""
        try:
            if store_id in self.openai_vector_stores:
                data = [{"store_id": store_id, "vector": [0], "metadata": json.dumps(store_info)}]
                await asyncio.to_thread(
                    self.client.upsert,
                    collection_name=self.metadata_collection_name,
                    data=data,
                )
                self.openai_vector_stores[store_id] = store_info
        except Exception as e:
            logger.error(f"error updating openai vector store {store_id}: {e}")
            raise

    async def _delete_openai_vector_store_from_storage(self, store_id: str) -> None:
        """Delete the OpenAI vector store from Milvus metadata collection."""
        try:
            if store_id in self.openai_vector_stores:
                if await asyncio.to_thread(self.client.has_collection, self.metadata_collection_name):
                    await asyncio.to_thread(
                        self.client.delete,
                        collection_name=self.metadata_collection_name,
                        filter=f"store_id in ['{store_id}']",
                    )
                # remove from in-memory cache
                del self.openai_vector_stores[store_id]

        except Exception as e:
            logger.error(f"error deleting openai vector store {store_id}: {e}")
            raise
