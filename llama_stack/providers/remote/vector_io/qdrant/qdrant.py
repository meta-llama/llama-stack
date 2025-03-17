# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from numpy.typing import NDArray
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct

from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse, VectorIO
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.inline.vector_io.qdrant import QdrantVectorIOConfig as InlineQdrantVectorIOConfig
from llama_stack.providers.utils.memory.vector_store import (
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import QdrantVectorIOConfig as RemoteQdrantVectorIOConfig

log = logging.getLogger(__name__)
CHUNK_ID_KEY = "_chunk_id"


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

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )

        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE),
            )

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
            chunk_id = f"{chunk.metadata['document_id']}:chunk-{i}"
            points.append(
                PointStruct(
                    id=convert_id(chunk_id),
                    vector=embedding,
                    payload={"chunk_content": chunk.model_dump()} | {CHUNK_ID_KEY: chunk_id},
                )
            )

        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def query(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
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

    async def delete(self):
        await self.client.delete_collection(collection_name=self.collection_name)


class QdrantVectorIOAdapter(VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self, config: Union[RemoteQdrantVectorIOConfig, InlineQdrantVectorIOConfig], inference_api: Api.inference
    ) -> None:
        self.config = config
        self.client: AsyncQdrantClient = None
        self.cache = {}
        self.inference_api = inference_api

    async def initialize(self) -> None:
        self.client = AsyncQdrantClient(**self.config.model_dump(exclude_none=True))

    async def shutdown(self) -> None:
        await self.client.close()

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

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> Optional[VectorDBWithIndex]:
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
