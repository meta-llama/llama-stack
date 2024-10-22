# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import traceback
import uuid
from typing import Any, Dict, List

from numpy.typing import NDArray
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import PointStruct

from llama_stack.providers.datatypes import MemoryBanksProtocolPrivate

from llama_stack.apis.memory import *  # noqa: F403

from llama_stack.providers.adapters.memory.qdrant.config import QdrantConfig
from llama_stack.providers.utils.memory.vector_store import (
    BankWithIndex,
    EmbeddingIndex,
)

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
        assert len(chunks) == len(
            embeddings
        ), f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"

        if not await self.client.collection_exists(self.collection_name):
            await self.client.create_collection(
                self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(embeddings[0]), distance=models.Distance.COSINE
                ),
            )

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{chunk.document_id}:chunk-{i}"
            points.append(
                PointStruct(
                    id=convert_id(chunk_id),
                    vector=embedding,
                    payload={"chunk_content": chunk.model_dump()}
                    | {CHUNK_ID_KEY: chunk_id},
                )
            )

        await self.client.upsert(collection_name=self.collection_name, points=points)

    async def query(
        self, embedding: NDArray, k: int, score_threshold: float
    ) -> QueryDocumentsResponse:
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
                traceback.print_exc()
                continue

            chunks.append(chunk)
            scores.append(point.score)

        return QueryDocumentsResponse(chunks=chunks, scores=scores)


class QdrantVectorMemoryAdapter(Memory, MemoryBanksProtocolPrivate):
    def __init__(self, config: QdrantConfig) -> None:
        self.config = config
        self.client = AsyncQdrantClient(**self.config.model_dump(exclude_none=True))
        self.cache = {}

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        self.client.close()

    async def register_memory_bank(
        self,
        memory_bank: MemoryBankDef,
    ) -> None:
        assert (
            memory_bank.type == MemoryBankType.vector.value
        ), f"Only vector banks are supported {memory_bank.type}"

        index = BankWithIndex(
            bank=memory_bank,
            index=QdrantIndex(self.client, memory_bank.identifier),
        )

        self.cache[memory_bank.identifier] = index

    async def list_memory_banks(self) -> List[MemoryBankDef]:
        # Qdrant doesn't have collection level metadata to store the bank properties
        # So we only return from the cache value
        return [i.bank for i in self.cache.values()]

    async def _get_and_cache_bank_index(self, bank_id: str) -> Optional[BankWithIndex]:
        if bank_id in self.cache:
            return self.cache[bank_id]

        bank = await self.memory_bank_store.get_memory_bank(bank_id)
        if not bank:
            raise ValueError(f"Bank {bank_id} not found")

        index = BankWithIndex(
            bank=bank,
            index=QdrantIndex(client=self.client, collection_name=bank_id),
        )
        self.cache[bank_id] = index
        return index

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = await self._get_and_cache_bank_index(bank_id)
        if not index:
            raise ValueError(f"Bank {bank_id} not found")

        await index.insert_documents(documents)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        index = await self._get_and_cache_bank_index(bank_id)
        if not index:
            raise ValueError(f"Bank {bank_id} not found")

        return await index.query_documents(query, params)
