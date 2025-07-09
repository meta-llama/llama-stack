# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
import os
import re
from typing import Any

from numpy.typing import NDArray
from pymilvus import AsyncMilvusClient, DataType

from llama_stack.apis.files.files import Files
from llama_stack.apis.inference import Inference, InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack.providers.datatypes import VectorDBsProtocolPrivate
from llama_stack.providers.inline.vector_io.milvus import MilvusVectorIOConfig as InlineMilvusVectorIOConfig
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import MilvusVectorIOConfig as RemoteMilvusVectorIOConfig

logger = logging.getLogger(__name__)

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_dbs:milvus:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:milvus:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:milvus:{VERSION}::"


def sanitize_collection_name(name: str) -> str:
    """
    Sanitize collection name to ensure it only contains numbers, letters, and underscores.
    Any other characters are replaced with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


class MilvusIndex(EmbeddingIndex):
    def __init__(
        self,
        client: AsyncMilvusClient,
        collection_name: str,
        consistency_level="Strong",
        kvstore: KVStore | None = None,
    ):
        self.client = client
        self.collection_name = sanitize_collection_name(collection_name)
        self.consistency_level = consistency_level
        self.kvstore = kvstore

    async def delete(self):
        if await self.client.collection_exists(self.collection_name):
            await self.client.drop_collection(collection_name=self.collection_name)

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )
        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                self.collection_name,
                dimension=len(embeddings[0]),
                auto_id=True,
                consistency_level=self.consistency_level,
            )

        data = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "vector": embedding,
                    "chunk_content": chunk.model_dump(),
                }
            )
        try:
            await self.client.insert(
                self.collection_name,
                data=data,
            )
        except Exception as e:
            logger.error(f"Error inserting chunks into Milvus collection {self.collection_name}: {e}")
            raise e

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        search_res = await self.client.search(
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
        self,
        config: RemoteMilvusVectorIOConfig | InlineMilvusVectorIOConfig,
        inference_api: Inference,
        files_api: Files | None,
    ) -> None:
        self.config = config
        self.cache = {}
        self.client: AsyncMilvusClient | None = None
        self.inference_api = inference_api
        self.files_api = files_api
        self.kvstore: KVStore | None = None
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        if self.config.kvstore is not None:
            self.kvstore = await kvstore_impl(self.config.kvstore)

        # Initialize client first before using it
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            logger.info(f"Connecting to Milvus server at {self.config.uri}")
            self.client = AsyncMilvusClient(**self.config.model_dump(exclude_none=True))
        else:
            logger.info(f"Connecting to Milvus Lite at: {self.config.db_path}")
            uri = os.path.expanduser(self.config.db_path)
            self.client = AsyncMilvusClient(uri=uri)

        # Now load stored vector databases
        if self.kvstore is not None:
            start_key = VECTOR_DBS_PREFIX
            end_key = f"{VECTOR_DBS_PREFIX}\xff"
            stored_vector_dbs = await self.kvstore.values_in_range(start_key, end_key)

            for vector_db_data in stored_vector_dbs:
                vector_db = VectorDB.model_validate_json(vector_db_data)
                index = VectorDBWithIndex(
                    vector_db,
                    index=MilvusIndex(
                        client=self.client,
                        collection_name=vector_db.identifier,
                        consistency_level=self.config.consistency_level,
                        kvstore=self.kvstore,
                    ),
                    inference_api=self.inference_api,
                )
                self.cache[vector_db.identifier] = index

        self.openai_vector_stores = await self._load_openai_vector_stores()

    async def shutdown(self) -> None:
        if self.client is not None:
            await self.client.close()

    def _ensure_client_initialized(self) -> AsyncMilvusClient:
        """Ensure the client is initialized and return it."""
        if self.client is None:
            raise RuntimeError("Milvus client is not initialized. Call initialize() first.")
        return self.client

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            consistency_level = self.config.consistency_level
        else:
            consistency_level = "Strong"
        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=MilvusIndex(
                self._ensure_client_initialized(), vector_db.identifier, consistency_level=consistency_level
            ),
            inference_api=self.inference_api,
        )

        self.cache[vector_db.identifier] = index

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex | None:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        # Vector DB should be registered before use
        raise ValueError(f"Vector DB {vector_db_id} not found. Please register it first.")

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
        """Save vector store metadata to persistent storage."""
        assert self.kvstore is not None
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.set(key=key, value=json.dumps(store_info))
        self.openai_vector_stores[store_id] = store_info

    async def _update_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Update vector store metadata in persistent storage."""
        assert self.kvstore is not None
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.set(key=key, value=json.dumps(store_info))
        self.openai_vector_stores[store_id] = store_info

    async def _delete_openai_vector_store_from_storage(self, store_id: str) -> None:
        """Delete vector store metadata from persistent storage."""
        assert self.kvstore is not None
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.delete(key)
        if store_id in self.openai_vector_stores:
            del self.openai_vector_stores[store_id]

    async def _load_openai_vector_stores(self) -> dict[str, dict[str, Any]]:
        """Load all vector store metadata from persistent storage."""
        assert self.kvstore is not None
        start_key = OPENAI_VECTOR_STORES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_PREFIX}\xff"
        stored = await self.kvstore.values_in_range(start_key, end_key)
        return {json.loads(s)["id"]: json.loads(s) for s in stored}

    async def _save_openai_vector_store_file(
        self, store_id: str, file_id: str, file_info: dict[str, Any], file_contents: list[dict[str, Any]]
    ) -> None:
        """Save vector store file metadata to Milvus database."""
        if store_id not in self.openai_vector_stores:
            # Reload all vector stores to check if the store exists
            self.openai_vector_stores = await self._load_openai_vector_stores()
            if store_id not in self.openai_vector_stores:
                logger.error(f"OpenAI vector store {store_id} not found")
                raise ValueError(f"No vector store found with id {store_id}")

        try:
            client = self._ensure_client_initialized()

            if not await client.collection_exists("openai_vector_store_files"):
                file_schema = AsyncMilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                    description="Metadata for OpenAI vector store files",
                )
                file_schema.add_field(
                    field_name="store_file_id", datatype=DataType.VARCHAR, is_primary=True, max_length=512
                )
                file_schema.add_field(field_name="store_id", datatype=DataType.VARCHAR, max_length=512)
                file_schema.add_field(field_name="file_id", datatype=DataType.VARCHAR, max_length=512)
                file_schema.add_field(field_name="file_info", datatype=DataType.VARCHAR, max_length=65535)

                await client.create_collection(
                    collection_name="openai_vector_store_files",
                    schema=file_schema,
                )

            if not await client.collection_exists("openai_vector_store_files_contents"):
                content_schema = AsyncMilvusClient.create_schema(
                    auto_id=False,
                    enable_dynamic_field=True,
                    description="Contents for OpenAI vector store files",
                )
                content_schema.add_field(
                    field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=1024
                )
                content_schema.add_field(field_name="store_file_id", datatype=DataType.VARCHAR, max_length=1024)
                content_schema.add_field(field_name="store_id", datatype=DataType.VARCHAR, max_length=512)
                content_schema.add_field(field_name="file_id", datatype=DataType.VARCHAR, max_length=512)
                content_schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)

                await client.create_collection(
                    collection_name="openai_vector_store_files_contents",
                    schema=content_schema,
                )

            file_data = [
                {
                    "store_file_id": f"{store_id}_{file_id}",
                    "store_id": store_id,
                    "file_id": file_id,
                    "file_info": json.dumps(file_info),
                }
            ]
            await client.upsert(
                collection_name="openai_vector_store_files",
                data=file_data,
            )

            # Save file contents
            contents_data = []
            for content in file_contents:
                chunk_metadata = content.get("chunk_metadata")
                if chunk_metadata and chunk_metadata.get("chunk_id"):
                    contents_data.append(
                        {
                            "chunk_id": chunk_metadata.get("chunk_id"),
                            "store_file_id": f"{store_id}_{file_id}",
                            "store_id": store_id,
                            "file_id": file_id,
                            "content": json.dumps(content),
                        }
                    )

            if contents_data:
                await client.upsert(collection_name="openai_vector_store_files_contents", data=contents_data)

        except Exception as e:
            logger.error(f"Error saving openai vector store file {file_id} for store {store_id}: {e}")
            raise

    async def _load_openai_vector_store_file(self, store_id: str, file_id: str) -> dict[str, Any]:
        """Load vector store file metadata from Milvus database."""
        try:
            client = self._ensure_client_initialized()

            if not await client.collection_exists("openai_vector_store_files"):
                return {}

            query_filter = f"store_file_id == '{store_id}_{file_id}'"
            results = await client.query(
                collection_name="openai_vector_store_files", filter=query_filter, output_fields=["file_info"]
            )

            if results:
                try:
                    return json.loads(results[0]["file_info"])
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode file_info for store {store_id}, file {file_id}: {e}")
                    return {}
            return {}
        except Exception as e:
            logger.error(f"Error loading openai vector store file {file_id} for store {store_id}: {e}")
            return {}

    async def _update_openai_vector_store_file(self, store_id: str, file_id: str, file_info: dict[str, Any]) -> None:
        """Update vector store file metadata in Milvus database."""
        try:
            client = self._ensure_client_initialized()

            if not await client.collection_exists("openai_vector_store_files"):
                return

            file_data = [
                {
                    "store_file_id": f"{store_id}_{file_id}",
                    "store_id": store_id,
                    "file_id": file_id,
                    "file_info": json.dumps(file_info),
                }
            ]
            await client.upsert(collection_name="openai_vector_store_files", data=file_data)
        except Exception as e:
            logger.error(f"Error updating openai vector store file {file_id} for store {store_id}: {e}")
            raise

    async def _load_openai_vector_store_file_contents(self, store_id: str, file_id: str) -> list[dict[str, Any]]:
        """Load vector store file contents from Milvus database."""
        try:
            client = self._ensure_client_initialized()

            if not await client.collection_exists("openai_vector_store_files_contents"):
                return []

            query_filter = (
                f"store_id == '{store_id}' AND file_id == '{file_id}' AND store_file_id == '{store_id}_{file_id}'"
            )
            results = await client.query(
                collection_name="openai_vector_store_files_contents",
                filter=query_filter,
                output_fields=["chunk_id", "store_id", "file_id", "content"],
            )

            contents = []
            for result in results:
                try:
                    content = json.loads(result["content"])
                    contents.append(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode content for store {store_id}, file {file_id}: {e}")
            return contents
        except Exception as e:
            logger.error(f"Error loading openai vector store file contents for {file_id} in store {store_id}: {e}")
            return []

    async def _delete_openai_vector_store_file_from_storage(self, store_id: str, file_id: str) -> None:
        """Delete vector store file metadata from Milvus database."""
        try:
            client = self._ensure_client_initialized()

            if not await client.collection_exists("openai_vector_store_files"):
                return

            query_filter = f"store_file_id in ['{store_id}_{file_id}']"
            await client.delete(collection_name="openai_vector_store_files", filter=query_filter)
            if await client.collection_exists("openai_vector_store_files_contents"):
                await client.delete(collection_name="openai_vector_store_files_contents", filter=query_filter)

        except Exception as e:
            logger.error(f"Error deleting openai vector store file {file_id} for store {store_id}: {e}")
            raise
