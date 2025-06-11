# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from typing import Any, Literal

import faiss
import numpy as np
from numpy.typing import NDArray

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.inference.inference import Inference
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    VectorIO,
    VectorStoreDeleteResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponse,
)
from llama_stack.providers.datatypes import VectorDBsProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import FaissVectorIOConfig

logger = logging.getLogger(__name__)

VERSION = "v3"
VECTOR_DBS_PREFIX = f"vector_dbs:{VERSION}::"
FAISS_INDEX_PREFIX = f"faiss_index:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:{VERSION}::"


# In faiss, since we do
CHUNK_MULTIPLIER = 5


class FaissIndex(EmbeddingIndex):
    def __init__(self, dimension: int, kvstore: KVStore | None = None, bank_id: str | None = None):
        self.index = faiss.IndexFlatL2(dimension)
        self.chunk_by_index: dict[int, Chunk] = {}
        self.kvstore = kvstore
        self.bank_id = bank_id

    @classmethod
    async def create(cls, dimension: int, kvstore: KVStore | None = None, bank_id: str | None = None):
        instance = cls(dimension, kvstore, bank_id)
        await instance.initialize()
        return instance

    async def initialize(self) -> None:
        if not self.kvstore:
            return

        index_key = f"{FAISS_INDEX_PREFIX}{self.bank_id}"
        stored_data = await self.kvstore.get(index_key)

        if stored_data:
            data = json.loads(stored_data)
            self.chunk_by_index = {int(k): Chunk.model_validate_json(v) for k, v in data["chunk_by_index"].items()}

            buffer = io.BytesIO(base64.b64decode(data["faiss_index"]))
            self.index = faiss.deserialize_index(np.loadtxt(buffer, dtype=np.uint8))

    async def _save_index(self):
        if not self.kvstore or not self.bank_id:
            return

        np_index = faiss.serialize_index(self.index)
        buffer = io.BytesIO()
        np.savetxt(buffer, np_index)
        data = {
            "chunk_by_index": {k: v.model_dump_json() for k, v in self.chunk_by_index.items()},
            "faiss_index": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        }

        index_key = f"{FAISS_INDEX_PREFIX}{self.bank_id}"
        await self.kvstore.set(key=index_key, value=json.dumps(data))

    async def delete(self):
        if not self.kvstore or not self.bank_id:
            return

        await self.kvstore.delete(f"{FAISS_INDEX_PREFIX}{self.bank_id}")

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        # Add dimension check
        embedding_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0]
        if embedding_dim != self.index.d:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.index.d}, got {embedding_dim}")

        indexlen = len(self.chunk_by_index)
        for i, chunk in enumerate(chunks):
            self.chunk_by_index[indexlen + i] = chunk

        self.index.add(np.array(embeddings).astype(np.float32))

        # Save updated index
        await self._save_index()

    async def query_vector(
        self,
        embedding: NDArray,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        distances, indices = await asyncio.to_thread(self.index.search, embedding.reshape(1, -1).astype(np.float32), k)
        chunks = []
        scores = []
        for d, i in zip(distances[0], indices[0], strict=False):
            if i < 0:
                continue
            chunks.append(self.chunk_by_index[int(i)])
            scores.append(1.0 / float(d) if d != 0 else float("inf"))

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Keyword search is not supported in FAISS")


class FaissVectorIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(self, config: FaissVectorIOConfig, inference_api: Inference) -> None:
        self.config = config
        self.inference_api = inference_api
        self.cache: dict[str, VectorDBWithIndex] = {}
        self.kvstore: KVStore | None = None
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing banks from kvstore
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_dbs = await self.kvstore.values_in_range(start_key, end_key)

        for vector_db_data in stored_vector_dbs:
            vector_db = VectorDB.model_validate_json(vector_db_data)
            index = VectorDBWithIndex(
                vector_db,
                await FaissIndex.create(vector_db.embedding_dimension, self.kvstore, vector_db.identifier),
                self.inference_api,
            )
            self.cache[vector_db.identifier] = index

        # Load existing OpenAI vector stores
        start_key = OPENAI_VECTOR_STORES_PREFIX
        end_key = f"{OPENAI_VECTOR_STORES_PREFIX}\xff"
        stored_openai_stores = await self.kvstore.values_in_range(start_key, end_key)

        for store_data in stored_openai_stores:
            store_info = json.loads(store_data)
            self.openai_vector_stores[store_info["id"]] = store_info

    async def shutdown(self) -> None:
        # Cleanup if needed
        pass

    async def register_vector_db(
        self,
        vector_db: VectorDB,
    ) -> None:
        assert self.kvstore is not None

        key = f"{VECTOR_DBS_PREFIX}{vector_db.identifier}"
        await self.kvstore.set(
            key=key,
            value=vector_db.model_dump_json(),
        )

        # Store in cache
        self.cache[vector_db.identifier] = VectorDBWithIndex(
            vector_db=vector_db,
            index=await FaissIndex.create(vector_db.embedding_dimension, self.kvstore, vector_db.identifier),
            inference_api=self.inference_api,
        )

    async def list_vector_dbs(self) -> list[VectorDB]:
        return [i.vector_db for i in self.cache.values()]

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        assert self.kvstore is not None

        if vector_db_id not in self.cache:
            logger.warning(f"Vector DB {vector_db_id} not found")
            return

        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]
        await self.kvstore.delete(f"{VECTOR_DBS_PREFIX}{vector_db_id}")

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        index = self.cache.get(vector_db_id)
        if index is None:
            raise ValueError(f"Vector DB {vector_db_id} not found. found: {self.cache.keys()}")

        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        index = self.cache.get(vector_db_id)
        if index is None:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        return await index.query_chunks(query, params)

    # OpenAI Vector Stores API endpoints implementation
    async def openai_create_vector_store(
        self,
        name: str | None = None,
        file_ids: list[str] | None = None,
        expires_after: dict[str, Any] | None = None,
        chunking_strategy: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = 384,
        provider_id: str | None = None,
        provider_vector_db_id: str | None = None,
    ) -> VectorStoreObject:
        """Creates a vector store."""
        assert self.kvstore is not None
        # store and vector_db have the same id
        store_id = name or str(uuid.uuid4())
        created_at = int(time.time())

        if provider_id is None:
            raise ValueError("Provider ID is required")

        if embedding_model is None:
            raise ValueError("Embedding model is required")

        # Use provided embedding dimension or default to 384
        if embedding_dimension is None:
            raise ValueError("Embedding dimension is required")

        provider_vector_db_id = provider_vector_db_id or store_id
        vector_db = VectorDB(
            identifier=store_id,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model,
            provider_id=provider_id,
            provider_resource_id=provider_vector_db_id,
        )

        # Register the vector DB
        await self.register_vector_db(vector_db)

        # Create OpenAI vector store metadata
        store_info = {
            "id": store_id,
            "object": "vector_store",
            "created_at": created_at,
            "name": store_id,
            "usage_bytes": 0,
            "file_counts": {},
            "status": "completed",
            "expires_after": expires_after,
            "expires_at": None,
            "last_active_at": created_at,
            "file_ids": file_ids or [],
            "chunking_strategy": chunking_strategy,
        }

        # Add provider information to metadata if provided
        metadata = metadata or {}
        if provider_id:
            metadata["provider_id"] = provider_id
        if provider_vector_db_id:
            metadata["provider_vector_db_id"] = provider_vector_db_id
        store_info["metadata"] = metadata

        # Store in kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{store_id}"
        await self.kvstore.set(key=key, value=json.dumps(store_info))

        # Store in memory cache
        self.openai_vector_stores[store_id] = store_info

        return VectorStoreObject(
            id=store_id,
            created_at=created_at,
            name=store_id,
            usage_bytes=0,
            file_counts={},
            status="completed",
            expires_after=expires_after,
            expires_at=None,
            last_active_at=created_at,
            metadata=metadata,
        )

    async def openai_list_vector_stores(
        self,
        limit: int = 20,
        order: str = "desc",
        after: str | None = None,
        before: str | None = None,
    ) -> VectorStoreListResponse:
        """Returns a list of vector stores."""
        # Get all vector stores
        all_stores = list(self.openai_vector_stores.values())

        # Sort by created_at
        reverse_order = order == "desc"
        all_stores.sort(key=lambda x: x["created_at"], reverse=reverse_order)

        # Apply cursor-based pagination
        if after:
            after_index = next((i for i, store in enumerate(all_stores) if store["id"] == after), -1)
            if after_index >= 0:
                all_stores = all_stores[after_index + 1 :]

        if before:
            before_index = next((i for i, store in enumerate(all_stores) if store["id"] == before), len(all_stores))
            all_stores = all_stores[:before_index]

        # Apply limit
        limited_stores = all_stores[:limit]
        # Convert to VectorStoreObject instances
        data = [VectorStoreObject(**store) for store in limited_stores]

        # Determine pagination info
        has_more = len(all_stores) > limit
        first_id = data[0].id if data else None
        last_id = data[-1].id if data else None

        return VectorStoreListResponse(
            data=data,
            has_more=has_more,
            first_id=first_id,
            last_id=last_id,
        )

    async def openai_retrieve_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreObject:
        """Retrieves a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise ValueError(f"Vector store {vector_store_id} not found")

        store_info = self.openai_vector_stores[vector_store_id]
        return VectorStoreObject(**store_info)

    async def openai_update_vector_store(
        self,
        vector_store_id: str,
        name: str | None = None,
        expires_after: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStoreObject:
        """Modifies a vector store."""
        assert self.kvstore is not None
        if vector_store_id not in self.openai_vector_stores:
            raise ValueError(f"Vector store {vector_store_id} not found")

        store_info = self.openai_vector_stores[vector_store_id].copy()

        # Update fields if provided
        if name is not None:
            store_info["name"] = name
        if expires_after is not None:
            store_info["expires_after"] = expires_after
        if metadata is not None:
            store_info["metadata"] = metadata

        # Update last_active_at
        store_info["last_active_at"] = int(time.time())

        # Save to kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{vector_store_id}"
        await self.kvstore.set(key=key, value=json.dumps(store_info))

        # Update in-memory cache
        self.openai_vector_stores[vector_store_id] = store_info

        return VectorStoreObject(**store_info)

    async def openai_delete_vector_store(
        self,
        vector_store_id: str,
    ) -> VectorStoreDeleteResponse:
        """Delete a vector store."""
        assert self.kvstore is not None
        if vector_store_id not in self.openai_vector_stores:
            raise ValueError(f"Vector store {vector_store_id} not found")

        # Delete from kvstore
        key = f"{OPENAI_VECTOR_STORES_PREFIX}{vector_store_id}"
        await self.kvstore.delete(key)

        # Delete from in-memory cache
        del self.openai_vector_stores[vector_store_id]

        # Also delete the underlying vector DB
        try:
            await self.unregister_vector_db(vector_store_id)
        except Exception as e:
            logger.warning(f"Failed to delete underlying vector DB {vector_store_id}: {e}")

        return VectorStoreDeleteResponse(
            id=vector_store_id,
            deleted=True,
        )

    async def openai_search_vector_store(
        self,
        vector_store_id: str,
        query: str | list[str],
        filters: dict[str, Any] | None = None,
        max_num_results: int = 10,
        ranking_options: dict[str, Any] | None = None,
        rewrite_query: bool = False,
        search_mode: Literal["keyword", "vector", "hybrid"] = "vector",
    ) -> VectorStoreSearchResponse:
        """Search for chunks in a vector store."""
        if vector_store_id not in self.openai_vector_stores:
            raise ValueError(f"Vector store {vector_store_id} not found")

        if isinstance(query, list):
            search_query = " ".join(query)
        else:
            search_query = query

        try:
            score_threshold = ranking_options.get("score_threshold", 0.0) if ranking_options else 0.0
            params = {
                "max_chunks": max_num_results * CHUNK_MULTIPLIER,
                "score_threshold": score_threshold,
                "mode": search_mode,
            }
            # TODO: Add support for ranking_options.ranker

            response = await self.query_chunks(
                vector_db_id=vector_store_id,
                query=search_query,
                params=params,
            )

            # Convert response to OpenAI format
            data = []
            for i, (chunk, score) in enumerate(zip(response.chunks, response.scores, strict=False)):
                # Apply score based filtering
                if score < score_threshold:
                    continue

                # Apply filters if provided
                if filters:
                    # Simple metadata filtering
                    if not self._matches_filters(chunk.metadata, filters):
                        continue

                chunk_data = {
                    "id": f"chunk_{i}",
                    "object": "vector_store.search_result",
                    "score": score,
                    "content": chunk.content.content if hasattr(chunk.content, "content") else str(chunk.content),
                    "metadata": chunk.metadata,
                }
                data.append(chunk_data)
                if len(data) >= max_num_results:
                    break

            return VectorStoreSearchResponse(
                search_query=search_query,
                data=data,
                has_more=False,  # For simplicity, we don't implement pagination here
                next_page=None,
            )

        except Exception as e:
            logger.error(f"Error searching vector store {vector_store_id}: {e}")
            # Return empty results on error
            return VectorStoreSearchResponse(
                search_query=search_query,
                data=[],
                has_more=False,
                next_page=None,
            )

    def _matches_filters(self, metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True
