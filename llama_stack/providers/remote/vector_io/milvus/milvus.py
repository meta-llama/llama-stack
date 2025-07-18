# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import logging
import os
import re
from typing import Any

from numpy.typing import NDArray
from pymilvus import DataType, Function, FunctionType, MilvusClient

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
        self, client: MilvusClient, collection_name: str, consistency_level="Strong", kvstore: KVStore | None = None
    ):
        self.client = client
        self.collection_name = sanitize_collection_name(collection_name)
        self.consistency_level = consistency_level
        self.kvstore = kvstore

    async def initialize(self):
        # MilvusIndex does not require explicit initialization
        # TODO: could move collection creation into initialization but it is not really necessary
        pass

    async def delete(self):
        if await asyncio.to_thread(self.client.has_collection, self.collection_name):
            await asyncio.to_thread(self.client.drop_collection, collection_name=self.collection_name)

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        if not await asyncio.to_thread(self.client.has_collection, self.collection_name):
            logger.info(f"Creating new collection {self.collection_name} with nullable sparse field")
            # Create schema for vector search
            schema = self.client.create_schema()
            schema.add_field(
                field_name="chunk_id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=100,
            )
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True,  # Enable text analysis for BM25
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=len(embeddings[0]),
            )
            schema.add_field(
                field_name="chunk_content",
                datatype=DataType.JSON,
            )
            # Add sparse vector field for BM25 (required by the function)
            schema.add_field(
                field_name="sparse",
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )

            # Create indexes
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="FLAT",
                metric_type="COSINE",
            )
            # Add index for sparse field (required by BM25 function)
            index_params.add_index(
                field_name="sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
            )

            # Add BM25 function for full-text search
            bm25_function = Function(
                name="text_bm25_emb",
                input_field_names=["content"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
            schema.add_function(bm25_function)

            await asyncio.to_thread(
                self.client.create_collection,
                self.collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self.consistency_level,
            )

        data = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            data.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "vector": embedding,
                    "chunk_content": chunk.model_dump(),
                    # sparse field will be handled by BM25 function automatically
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
            anns_field="vector",
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
        """
        Perform BM25-based keyword search using Milvus's built-in full-text search.
        """
        try:
            # Use Milvus's built-in BM25 search
            search_res = await asyncio.to_thread(
                self.client.search,
                collection_name=self.collection_name,
                data=[query_string],  # Raw text query
                anns_field="sparse",  # Use sparse field for BM25
                output_fields=["chunk_content"],  # Output the chunk content
                limit=k,
                search_params={
                    "params": {
                        "drop_ratio_search": 0.2,  # Ignore low-importance terms
                    }
                },
            )

            chunks = []
            scores = []
            for res in search_res[0]:
                chunk = Chunk(**res["entity"]["chunk_content"])
                chunks.append(chunk)
                scores.append(res["distance"])  # BM25 score from Milvus

            # Filter by score threshold
            filtered_chunks = [chunk for chunk, score in zip(chunks, scores, strict=False) if score >= score_threshold]
            filtered_scores = [score for score in scores if score >= score_threshold]

            return QueryChunksResponse(chunks=filtered_chunks, scores=filtered_scores)

        except Exception as e:
            logger.error(f"Error performing BM25 search: {e}")
            # Fallback to simple text search
            return await self._fallback_keyword_search(query_string, k, score_threshold)

    async def _fallback_keyword_search(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        """
        Fallback to simple text search when BM25 search is not available.
        """
        # Simple text search using content field
        search_res = await asyncio.to_thread(
            self.client.query,
            collection_name=self.collection_name,
            filter='content like "%{content}%"',
            filter_params={"content": query_string},
            output_fields=["*"],
            limit=k,
        )
        chunks = [Chunk(**res["chunk_content"]) for res in search_res]
        scores = [1.0] * len(chunks)  # Simple binary score for text search
        return QueryChunksResponse(chunks=chunks, scores=scores)

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
        self.client = None
        self.inference_api = inference_api
        self.files_api = files_api
        self.kvstore: KVStore | None = None
        self.vector_db_store = None
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        # MilvusVectorIOAdapter is used for both inline and remote connections
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            # Remote Milvus: kvstore is optional for registry persistence across server restarts
            if self.config.kvstore is not None:
                self.kvstore = await kvstore_impl(self.config.kvstore)
                logger.info("Remote Milvus: Using kvstore for vector database registry persistence")
            else:
                self.kvstore = None
                logger.info("Remote Milvus: No kvstore configured, registry will not persist across restarts")
            if self.kvstore is not None:
                start_key = VECTOR_DBS_PREFIX
                end_key = f"{VECTOR_DBS_PREFIX}\xff"
                stored_vector_dbs = await self.kvstore.values_in_range(start_key, end_key)
            else:
                stored_vector_dbs = []

        elif isinstance(self.config, InlineMilvusVectorIOConfig):
            self.kvstore = await kvstore_impl(self.config.kvstore)
            logger.info("Inline Milvus: Using kvstore for local vector database registry")
            start_key = VECTOR_DBS_PREFIX
            end_key = f"{VECTOR_DBS_PREFIX}\xff"
            stored_vector_dbs = await self.kvstore.values_in_range(start_key, end_key)
        else:
            raise ValueError(
                f"Unsupported config type: {type(self.config)}. Expected RemoteMilvusVectorIOConfig or InlineMilvusVectorIOConfig"
            )

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
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            logger.info(f"Connecting to remote Milvus server at {self.config.uri}")
            self.client = MilvusClient(**self.config.model_dump(exclude_none=True))
        elif isinstance(self.config, InlineMilvusVectorIOConfig):
            logger.info(f"Connecting to local Milvus Lite at: {self.config.db_path}")
            uri = os.path.expanduser(self.config.db_path)
            self.client = MilvusClient(uri=uri)
        else:
            raise ValueError(
                f"Unsupported config type: {type(self.config)}. Expected RemoteMilvusVectorIOConfig or InlineMilvusVectorIOConfig"
            )

        # Load existing OpenAI vector stores into the in-memory cache
        await self.initialize_openai_vector_stores()

    async def shutdown(self) -> None:
        self.client.close()

    async def register_vector_db(
        self,
        vector_db: VectorDB,
    ) -> None:
        # Set consistency level based on configuration type
        if isinstance(self.config, RemoteMilvusVectorIOConfig):
            consistency_level = self.config.consistency_level
        elif isinstance(self.config, InlineMilvusVectorIOConfig):
            consistency_level = self.config.consistency_level
        else:
            raise ValueError(
                f"Unsupported config type: {type(self.config)}. Expected RemoteMilvusVectorIOConfig or InlineMilvusVectorIOConfig"
            )
        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=MilvusIndex(self.client, vector_db.identifier, consistency_level=consistency_level),
            inference_api=self.inference_api,
        )

        self.cache[vector_db.identifier] = index

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex | None:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        if self.vector_db_store is None:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        if not vector_db:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=MilvusIndex(client=self.client, collection_name=vector_db.identifier, kvstore=self.kvstore),
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

        if params and params.get("mode") == "keyword":
            # Check if this is inline Milvus (Milvus-Lite)
            if hasattr(self.config, "db_path"):
                raise NotImplementedError(
                    "Keyword search is not supported in Milvus-Lite. "
                    "Please use a remote Milvus server for keyword search functionality."
                )

        return await index.query_chunks(query, params)

    async def _save_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Save vector store metadata to persistent storage."""
        if self.kvstore is not None:
            key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
            await self.kvstore.set(key=key, value=json.dumps(store_info))
        self.openai_vector_stores[store_id] = store_info

    async def _update_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Update vector store metadata in persistent storage."""
        if self.kvstore is not None:
            key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
            await self.kvstore.set(key=key, value=json.dumps(store_info))
        self.openai_vector_stores[store_id] = store_info

    async def _delete_openai_vector_store_from_storage(self, store_id: str) -> None:
        """Delete vector store metadata from persistent storage."""
        if self.kvstore is not None:
            key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
            await self.kvstore.delete(key)
        if store_id in self.openai_vector_stores:
            del self.openai_vector_stores[store_id]

    async def _load_openai_vector_stores(self) -> dict[str, dict[str, Any]]:
        """Load all vector store metadata from persistent storage."""
        if self.kvstore is None:
            return {}
        start_key = OPENAI_VECTOR_STORES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_PREFIX}\xff"
        stored = await self.kvstore.values_in_range(start_key, end_key)
        return {json.loads(s)["id"]: json.loads(s) for s in stored}

    async def _save_openai_vector_store_file(
        self, store_id: str, file_id: str, file_info: dict[str, Any], file_contents: list[dict[str, Any]]
    ) -> None:
        """Save vector store file metadata to Milvus database."""
        if store_id not in self.openai_vector_stores:
            store_info = await self._load_openai_vector_stores(store_id)
            if not store_info:
                logger.error(f"OpenAI vector store {store_id} not found")
                raise ValueError(f"No vector store found with id {store_id}")

        try:
            if not await asyncio.to_thread(self.client.has_collection, "openai_vector_store_files"):
                file_schema = MilvusClient.create_schema(
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

                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name="openai_vector_store_files",
                    schema=file_schema,
                )

            if not await asyncio.to_thread(self.client.has_collection, "openai_vector_store_files_contents"):
                content_schema = MilvusClient.create_schema(
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

                await asyncio.to_thread(
                    self.client.create_collection,
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
            await asyncio.to_thread(
                self.client.upsert,
                collection_name="openai_vector_store_files",
                data=file_data,
            )

            # Save file contents
            contents_data = [
                {
                    "chunk_id": content.get("chunk_metadata").get("chunk_id"),
                    "store_file_id": f"{store_id}_{file_id}",
                    "store_id": store_id,
                    "file_id": file_id,
                    "content": json.dumps(content),
                }
                for content in file_contents
            ]
            await asyncio.to_thread(
                self.client.upsert,
                collection_name="openai_vector_store_files_contents",
                data=contents_data,
            )

        except Exception as e:
            logger.error(f"Error saving openai vector store file {file_id} for store {store_id}: {e}")

    async def _load_openai_vector_store_file(self, store_id: str, file_id: str) -> dict[str, Any]:
        """Load vector store file metadata from Milvus database."""
        try:
            if not await asyncio.to_thread(self.client.has_collection, "openai_vector_store_files"):
                return {}

            query_filter = f"store_file_id == '{store_id}_{file_id}'"
            results = await asyncio.to_thread(
                self.client.query,
                collection_name="openai_vector_store_files",
                filter=query_filter,
                output_fields=["file_info"],
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
            if not await asyncio.to_thread(self.client.has_collection, "openai_vector_store_files"):
                return

            file_data = [
                {
                    "store_file_id": f"{store_id}_{file_id}",
                    "store_id": store_id,
                    "file_id": file_id,
                    "file_info": json.dumps(file_info),
                }
            ]
            await asyncio.to_thread(
                self.client.upsert,
                collection_name="openai_vector_store_files",
                data=file_data,
            )
        except Exception as e:
            logger.error(f"Error updating openai vector store file {file_id} for store {store_id}: {e}")
            raise

    async def _load_openai_vector_store_file_contents(self, store_id: str, file_id: str) -> list[dict[str, Any]]:
        """Load vector store file contents from Milvus database."""
        try:
            if not await asyncio.to_thread(self.client.has_collection, "openai_vector_store_files_contents"):
                return []

            query_filter = (
                f"store_id == '{store_id}' AND file_id == '{file_id}' AND store_file_id == '{store_id}_{file_id}'"
            )
            results = await asyncio.to_thread(
                self.client.query,
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
            if not await asyncio.to_thread(self.client.has_collection, "openai_vector_store_files"):
                return

            query_filter = f"store_file_id in ['{store_id}_{file_id}']"
            await asyncio.to_thread(
                self.client.delete,
                collection_name="openai_vector_store_files",
                filter=query_filter,
            )
            if await asyncio.to_thread(self.client.has_collection, "openai_vector_store_files_contents"):
                await asyncio.to_thread(
                    self.client.delete,
                    collection_name="openai_vector_store_files_contents",
                    filter=query_filter,
                )

        except Exception as e:
            logger.error(f"Error deleting openai vector store file {file_id} for store {store_id}: {e}")
            raise
