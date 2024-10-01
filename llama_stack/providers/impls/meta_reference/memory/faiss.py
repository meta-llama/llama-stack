# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import uuid

from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from numpy.typing import NDArray

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.distribution.datatypes import RoutableProvider

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.providers.utils.memory.vector_store import (
    ALL_MINILM_L6_V2_DIMENSION,
    BankWithIndex,
    EmbeddingIndex,
)
from llama_stack.providers.utils.telemetry import tracing

from .config import FaissImplConfig

logger = logging.getLogger(__name__)


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

    async def query(self, embedding: NDArray, k: int) -> QueryDocumentsResponse:
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


class FaissMemoryImpl(Memory, RoutableProvider):
    def __init__(self, config: FaissImplConfig) -> None:
        self.config = config
        self.cache = {}

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    async def validate_routing_keys(self, routing_keys: List[str]) -> None:
        print(f"[faiss] Registering memory bank routing keys: {routing_keys}")
        pass

    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank:
        assert url is None, "URL is not supported for this implementation"
        assert (
            config.type == MemoryBankType.vector.value
        ), f"Only vector banks are supported {config.type}"

        bank_id = str(uuid.uuid4())
        bank = MemoryBank(
            bank_id=bank_id,
            name=name,
            config=config,
            url=url,
        )
        index = BankWithIndex(bank=bank, index=FaissIndex(ALL_MINILM_L6_V2_DIMENSION))
        self.cache[bank_id] = index
        return bank

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        index = self.cache.get(bank_id)
        if index is None:
            return None
        return index.bank

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
