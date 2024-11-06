# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging

from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from numpy.typing import NDArray

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.providers.datatypes import MemoryBanksProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl

from llama_stack.providers.utils.memory.vector_store import (
    ALL_MINILM_L6_V2_DIMENSION,
    BankWithIndex,
    EmbeddingIndex,
)
from llama_stack.providers.utils.telemetry import tracing

from .config import FaissImplConfig

logger = logging.getLogger(__name__)

MEMORY_BANKS_PREFIX = "memory_banks:"


class FaissIndex(EmbeddingIndex):
    id_by_index: Dict[int, str]
    chunk_by_index: Dict[int, str]

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.id_by_index = {}
        self.chunk_by_index = {}

    @tracing.span(name="add_chunks")
    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        indexlen = len(self.id_by_index)
        for i, chunk in enumerate(chunks):
            self.chunk_by_index[indexlen + i] = chunk
            self.id_by_index[indexlen + i] = chunk.document_id

        self.index.add(np.array(embeddings).astype(np.float32))

    async def query(
        self, embedding: NDArray, k: int, score_threshold: float
    ) -> QueryDocumentsResponse:
        distances, indices = self.index.search(
            embedding.reshape(1, -1).astype(np.float32), k
        )

        chunks = []
        scores = []
        for d, i in zip(distances[0], indices[0]):
            if i < 0:
                continue
            chunks.append(self.chunk_by_index[int(i)])
            scores.append(1.0 / float(d))

        return QueryDocumentsResponse(chunks=chunks, scores=scores)


class FaissMemoryImpl(Memory, MemoryBanksProtocolPrivate):
    def __init__(self, config: FaissImplConfig) -> None:
        self.config = config
        self.cache = {}
        self.kvstore = None

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)
        # Load existing banks from kvstore
        start_key = MEMORY_BANKS_PREFIX
        end_key = f"{MEMORY_BANKS_PREFIX}\xff"
        stored_banks = await self.kvstore.range(start_key, end_key)

        for bank_data in stored_banks:
            bank = VectorMemoryBankDef.model_validate_json(bank_data)
            index = BankWithIndex(
                bank=bank, index=FaissIndex(ALL_MINILM_L6_V2_DIMENSION)
            )
            self.cache[bank.identifier] = index

    async def shutdown(self) -> None:
        # Cleanup if needed
        pass

    async def register_memory_bank(
        self,
        memory_bank: MemoryBankDef,
    ) -> None:
        assert (
            memory_bank.type == MemoryBankType.vector.value
        ), f"Only vector banks are supported {memory_bank.type}"

        # Store in kvstore
        key = f"{MEMORY_BANKS_PREFIX}{memory_bank.identifier}"
        await self.kvstore.set(
            key=key,
            value=memory_bank.json(),
        )

        # Store in cache
        index = BankWithIndex(
            bank=memory_bank, index=FaissIndex(ALL_MINILM_L6_V2_DIMENSION)
        )
        self.cache[memory_bank.identifier] = index

    async def list_memory_banks(self) -> List[MemoryBankDef]:
        return [i.bank for i in self.cache.values()]

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = self.cache.get(bank_id)
        if index is None:
            raise ValueError(f"Bank {bank_id} not found")

        await index.insert_documents(documents)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        index = self.cache.get(bank_id)
        if index is None:
            raise ValueError(f"Bank {bank_id} not found")

        return await index.query_documents(query, params)
