# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import uuid
from typing import Any

from numpy.typing import NDArray
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct

from llama_stack.apis.files import Files
from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.inline.vector_io.qdrant import QdrantVectorIOConfig as InlineQdrantVectorIOConfig
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import QdrantVectorIOConfig as RemoteQdrantVectorIOConfig

log = logging.getLogger(__name__)
CHUNK_ID_KEY = "_chunk_id"
OPENAI_VECTOR_STORES_METADATA_COLLECTION = "openai_vector_stores_metadata"


def convert_id(_id: str) -> str:
    """
    Converts any string into a UUID string based on a seed.

    Qdrant accepts UUID strings and unsigned integers as point ID.
    We use a seed to convert each string into a UUID string deterministically.
    This allows us to overwrite the same point with the original ID.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, _id))


class QdrantIndex(EmbeddingIndex):
    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name
        self._distance_metric = None  # Will be set when collection is created

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray, metadata: dict[str, Any] | None = None):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        # Extract distance_metric from metadata if provided, default to COSINE
        distance_metric = "COSINE"  # Default
        if metadata is not None and "distance_metric" in metadata:
            distance_metric = metadata["distance_metric"]

        if not await self.client.collection_exists(self.collection_name):
            # Create collection with the specified distance metric
            distance = getattr(models.Distance, distance_metric, models.Distance.COSINE)
            self._distance_metric = distance_metric

            await self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(size=len(embeddings[0]), distance=distance),
            )
        else:
            # Collection already exists, warn if different distance metric was requested
            if self._distance_metric is None:
                # For now, assume COSINE as default since we can't easily extract it from collection info
                self._distance_metric = "COSINE"

            if self._distance_metric != distance_metric:
                log.warning(
                    f"Collection {self.collection_name} was created with distance metric '{self._distance_metric}', "
                    f"but '{distance_metric}' was requested. Using existing distance metric."
                )

        points = []
        for _i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            chunk_id = chunk.chunk_id
            points.append(
                PointStruct(
                    id=convert_id(chunk_id),
                    vector=embedding,
                    payload={"chunk_content": chunk.model_dump()} | {CHUNK_ID_KEY: chunk_id},
                )
            )

        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        # Distance metric is set at collection creation and cannot be changed
        results = (
            await self.client.query_points(
                collection_name=self.collection_name,
                query=embedding.tolist(),
                limit=k,
                with_payload=True,
                score_threshold=score_threshold,
            )
        ).points

        chunks, scores = [], []
        for point in results:
            assert isinstance(point, models.ScoredPoint)
            assert point.payload is not None

            try:
                chunk = Chunk(**point.payload["chunk_content"])
            except Exception:
                log.exception("Failed to parse chunk")
                continue

            chunks.append(chunk)
            scores.append(point.score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Keyword search is not supported in Qdrant")

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        raise NotImplementedError("Hybrid search is not supported in Qdrant")

    async def delete(self):
        await self.client.delete_collection(collection_name=self.collection_name)


class QdrantVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self,
        config: RemoteQdrantVectorIOConfig | InlineQdrantVectorIOConfig,
        inference_api: Api.inference,
        files_api: Files | None,
    ) -> None:
        self.config = config
        self.client: AsyncQdrantClient = None
        self.cache = {}
        self.inference_api = inference_api
        self.files_api = files_api
        self.vector_db_store = None
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        self.client = AsyncQdrantClient(**self.config.model_dump(exclude_none=True))
        # Load existing OpenAI vector stores using the mixin method
        self.openai_vector_stores = await self._load_openai_vector_stores()

    async def shutdown(self) -> None:
        await self.client.close()

    # OpenAI Vector Store Mixin abstract method implementations
    async def _save_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Save vector store metadata to Qdrant collection metadata."""
        # Store metadata in a special collection for vector store metadata
        metadata_collection = OPENAI_VECTOR_STORES_METADATA_COLLECTION

        # Create metadata collection if it doesn't exist
        if not await self.client.collection_exists(metadata_collection):
            # Use default distance metric for metadata collection
            distance = models.Distance.COSINE

            await self.client.create_collection(
                collection_name=metadata_collection,
                vectors_config=models.VectorParams(size=1, distance=distance),
            )

        # Store metadata as a point with dummy vector
        await self.client.upsert(
            collection_name=metadata_collection,
            points=[
                models.PointStruct(
                    id=convert_id(store_id),
                    vector=[0.0],  # Dummy vector
                    payload={"metadata": store_info},
                )
            ],
        )

    async def _load_openai_vector_stores(self) -> dict[str, dict[str, Any]]:
        """Load all vector store metadata from Qdrant."""
        metadata_collection = OPENAI_VECTOR_STORES_METADATA_COLLECTION

        if not await self.client.collection_exists(metadata_collection):
            return {}

        # Get all points from metadata collection
        points = await self.client.scroll(
            collection_name=metadata_collection,
            limit=1000,  # Reasonable limit for metadata
            with_payload=True,
        )

        stores = {}
        for point in points[0]:  # points[0] contains the actual points
            if point.payload and "metadata" in point.payload:
                store_info = point.payload["metadata"]
                stores[store_info["id"]] = store_info

        return stores

    async def _update_openai_vector_store(self, store_id: str, store_info: dict[str, Any]) -> None:
        """Update vector store metadata in Qdrant."""
        await self._save_openai_vector_store(store_id, store_info)

    async def _delete_openai_vector_store_from_storage(self, store_id: str) -> None:
        """Delete vector store metadata from Qdrant."""
        metadata_collection = OPENAI_VECTOR_STORES_METADATA_COLLECTION

        if await self.client.collection_exists(metadata_collection):
            await self.client.delete(
                collection_name=metadata_collection, points_selector=models.PointIdsList(points=[convert_id(store_id)])
            )

    async def _save_openai_vector_store_file(
        self, store_id: str, file_id: str, file_info: dict[str, Any], file_contents: list[dict[str, Any]]
    ) -> None:
        """Save vector store file metadata to Qdrant collection metadata."""
        # Store file metadata in a special collection for vector store file metadata
        file_metadata_collection = f"{OPENAI_VECTOR_STORES_METADATA_COLLECTION}_files"

        # Create file metadata collection if it doesn't exist
        if not await self.client.collection_exists(file_metadata_collection):
            distance = models.Distance.COSINE
            await self.client.create_collection(
                collection_name=file_metadata_collection,
                vectors_config=models.VectorParams(size=1, distance=distance),
            )

        # Store file metadata as a point with dummy vector
        file_key = f"{store_id}:{file_id}"
        await self.client.upsert(
            collection_name=file_metadata_collection,
            points=[
                models.PointStruct(
                    id=convert_id(file_key),
                    vector=[0.0],  # Dummy vector
                    payload={"file_info": file_info, "file_contents": file_contents},
                )
            ],
        )

    async def _load_openai_vector_store_file(self, store_id: str, file_id: str) -> dict[str, Any]:
        """Load vector store file metadata from Qdrant."""
        file_metadata_collection = f"{OPENAI_VECTOR_STORES_METADATA_COLLECTION}_files"

        if not await self.client.collection_exists(file_metadata_collection):
            return {}

        file_key = f"{store_id}:{file_id}"
        points = await self.client.retrieve(
            collection_name=file_metadata_collection,
            ids=[convert_id(file_key)],
            with_payload=True,
        )

        if points and points[0].payload and "file_info" in points[0].payload:
            return points[0].payload["file_info"]
        return {}

    async def _load_openai_vector_store_file_contents(self, store_id: str, file_id: str) -> list[dict[str, Any]]:
        """Load vector store file contents from Qdrant."""
        file_metadata_collection = f"{OPENAI_VECTOR_STORES_METADATA_COLLECTION}_files"

        if not await self.client.collection_exists(file_metadata_collection):
            return []

        file_key = f"{store_id}:{file_id}"
        points = await self.client.retrieve(
            collection_name=file_metadata_collection,
            ids=[convert_id(file_key)],
            with_payload=True,
        )

        if points and points[0].payload and "file_contents" in points[0].payload:
            return points[0].payload["file_contents"]
        return []

    async def _update_openai_vector_store_file(self, store_id: str, file_id: str, file_info: dict[str, Any]) -> None:
        """Update vector store file metadata in Qdrant."""
        file_metadata_collection = f"{OPENAI_VECTOR_STORES_METADATA_COLLECTION}_files"

        if not await self.client.collection_exists(file_metadata_collection):
            return

        # Get existing file contents
        existing_contents = await self._load_openai_vector_store_file_contents(store_id, file_id)

        # Update with new file info but keep existing contents
        await self._save_openai_vector_store_file(store_id, file_id, file_info, existing_contents)

    async def _delete_openai_vector_store_file_from_storage(self, store_id: str, file_id: str) -> None:
        """Delete vector store file metadata from Qdrant."""
        file_metadata_collection = f"{OPENAI_VECTOR_STORES_METADATA_COLLECTION}_files"

        if await self.client.collection_exists(file_metadata_collection):
            file_key = f"{store_id}:{file_id}"
            await self.client.delete(
                collection_name=file_metadata_collection,
                points_selector=models.PointIdsList(points=[convert_id(file_key)]),
            )

    async def register_vector_db(
        self,
        vector_db: VectorDB,
    ) -> None:
        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=QdrantIndex(self.client, vector_db.identifier),
            inference_api=self.inference_api,
        )

        self.cache[vector_db.identifier] = index

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        if vector_db_id in self.cache:
            await self.cache[vector_db_id].index.delete()
            del self.cache[vector_db_id]

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex | None:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        if not vector_db:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        index = VectorDBWithIndex(
            vector_db=vector_db,
            index=QdrantIndex(client=self.client, collection_name=vector_db.identifier),
            inference_api=self.inference_api,
        )
        self.cache[vector_db_id] = index
        return index

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        if not index:
            raise ValueError(f"Vector DB {vector_db_id} not found")

        # Extract distance_metric from params if provided
        distance_metric = None
        if params is not None:
            distance_metric = params.get("distance_metric")

        await index.insert_chunks(chunks, distance_metric=distance_metric)

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
