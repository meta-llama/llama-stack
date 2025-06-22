# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import json
import logging
import uuid
from typing import Any
from urllib.parse import urlparse

import chromadb
from chromadb.errors import NotFoundError
from numpy.typing import NDArray

from llama_stack.apis.files import Files
from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    SearchRankingOptions,
    VectorIO,
    VectorStoreDeleteResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
    VectorStoreFileDeleteResponse,
)
from llama_stack.apis.vector_io.vector_io import (
    VectorStoreChunkingStrategy,
    VectorStoreDeleteResponse,
    VectorStoreFileContentsResponse,
    VectorStoreFileObject,
    VectorStoreFileStatus,
    VectorStoreListFilesResponse,
    VectorStoreListResponse,
    VectorStoreObject,
    VectorStoreSearchResponsePage,
)
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.inline.vector_io.chroma import ChromaVectorIOConfig as InlineChromaVectorIOConfig
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import ChromaVectorIOConfig as RemoteChromaVectorIOConfig

log = logging.getLogger(__name__)

ChromaClientType = chromadb.api.AsyncClientAPI | chromadb.api.ClientAPI


# this is a helper to allow us to use async and non-async chroma clients interchangeably
async def maybe_await(result):
    if asyncio.iscoroutine(result):
        return await result
    return result


class ChromaIndex(EmbeddingIndex):
    def __init__(self, client: ChromaClientType, collection):
        self.client = client
        self.collection = collection

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
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
        raise NotImplementedError("Keyword search is not supported in Chroma")

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Hybrid search is not supported in Chroma")


class ChromaVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self,
        config: RemoteChromaVectorIOConfig | InlineChromaVectorIOConfig,
        inference_api: Api.inference,
        files_api: Files | None
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
        await self.cache[vector_db_id].index.delete()
        del self.cache[vector_db_id]

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)

        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
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


    async def _save_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        try:
            collection = await maybe_await(self.client.get_collection(name=self.metadata_collection_name))
        except NotFoundError:
            collection = await maybe_await(
                self.client.create_collection(name=self.metadata_collection_name, metadata={
                    "description": "Collection to store metadata for OpenAI vector stores"
                })
            )

            await maybe_await(
                collection.add(
                    ids=[store_id],
                    metadatas=[{"store_id": store_id, "metadata": json.dumps(store_info)}],
                )
            )

            self.openai_vector_stores[store_id] = store_info

        except Exception as e:
            log.error(f"Error saving openai vector store {store_id}: {e}")
            raise

    async def _load_openai_vector_stores(self) -> dict[str, dict[str, Any]]:
        openai_vector_stores = {}
        try:
            collection = await maybe_await(self.client.get_collection(name=self.metadata_collection_name))
        except NotFoundError:
            return openai_vector_stores

        try:
            collection_count = await maybe_await(collection.count())
            if collection_count == 0:
                return openai_vector_stores
            offset = 0
            batch_size = 100
            while True:
                result = await maybe_await(
                    collection.get(
                        where={"store_id": {"$exists": True}},
                        offset=offset,
                        limit=batch_size,
                        include=["documents", "metadatas"],
                    )
                )
                if not result['ids'] or len(result['ids']) == 0:
                    break

                for i, doc_id in enumerate(result['ids']):
                    metadata = result.get('metadatas', [{}])[i] if i < len(result.get('metadatas', [])) else {}

                    # Extract store_id (assuming it's in metadata)
                    store_id = metadata.get('store_id')

                    if store_id:
                        # If metadata contains JSON string, parse it
                        metadata_json = metadata.get('metadata')
                        if metadata_json:
                            try:
                                if isinstance(metadata_json, str):
                                    store_info = json.loads(metadata_json)
                                else:
                                    store_info = metadata_json
                                openai_vector_stores[store_id] = store_info
                            except json.JSONDecodeError:
                                log.error(f"failed to decode metadata for store_id {store_id}")
                offset += batch_size
        except Exception as e:
            log.error(f"error loading openai vector stores: {e}")
        return openai_vector_stores

    async def _update_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        try:
            if store_id in self.openai_vector_stores:
                collection = await maybe_await(self.client.get_collection(name=self.metadata_collection_name))
                await maybe_await(
                    collection.update(
                        ids=[store_id],
                        metadatas=[{"store_id": store_id, "metadata": json.dumps(store_info)}],
                    )
                )
                self.openai_vector_stores[store_id] = store_info
        except NotFoundError:
            log.error(f"Collection {self.metadata_collection_name} not found")
        except Exception as e:
            log.error(f"Error updating openai vector store {store_id}: {e}")
            raise

    async def _delete_openai_vector_store_from_storage(self, store_id: str) -> None:
        try:
            collection = await maybe_await(self.client.get_collection(name=self.metadata_collection_name))
            await maybe_await(collection.delete(ids=[store_id]))
        except ValueError:
            log.error(f"Collection {self.metadata_collection_name} not found")
        except Exception as e:
            log.error(f"Error deleting openai vector store {store_id}: {e}")
            raise

    async def _delete_openai_vector_store_file_from_storage(self, store_id: str, file_id: str) -> None:
        """Delete vector store file metadata from persistent storage."""
    async def openai_list_files_in_vector_store(
        self,
        vector_store_id: str,
        limit: int | None = 20,
        order: str | None = "desc",
        after: str | None = None,
        before: str | None = None,
        filter: VectorStoreFileStatus | None = None,
    ) -> VectorStoreListFilesResponse:
        raise NotImplementedError("OpenAI Vector Stores API is not supported in Chroma")

    async def _load_openai_vector_store_file(self, store_id: str, file_id: str) -> dict[str, Any]:
        """Load vector store file metadata from persistent storage."""
        raise NotImplementedError("OpenAI Vector Stores API is not supported in Chroma")

    async def _load_openai_vector_store_file_contents(self, store_id: str, file_id: str) -> list[dict[str, Any]]:
        """Load vector store file contents from persistent storage."""
        raise NotImplementedError("OpenAI Vector Stores API is not supported in Chroma")

    async def _save_openai_vector_store_file(
        self, store_id: str, file_id: str, file_info: dict[str, Any], file_contents: list[dict[str, Any]]
    ) -> None:
        """Save vector store file metadata to persistent storage."""
        raise NotImplementedError("OpenAI Vector Stores API is not supported in Chroma")

    async def _update_openai_vector_store_file(self, store_id: str, file_id: str, file_info: dict[str, Any]) -> None:
        """Update vector store file metadata in persistent storage."""
        raise NotImplementedError("OpenAI Vector Stores API is not supported in Chroma")