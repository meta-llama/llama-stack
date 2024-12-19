# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import json
import logging
from typing import List, Optional, Union
from urllib.parse import urlparse

import chromadb
from numpy.typing import NDArray

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.memory_banks import MemoryBankType
from llama_stack.providers.datatypes import Api, MemoryBanksProtocolPrivate
from llama_stack.providers.inline.memory.chroma import ChromaInlineImplConfig
from llama_stack.providers.utils.memory.vector_store import (
    BankWithIndex,
    EmbeddingIndex,
)
from .config import ChromaRemoteImplConfig

log = logging.getLogger(__name__)


ChromaClientType = Union[chromadb.AsyncHttpClient, chromadb.PersistentClient]


# this is a helper to allow us to use async and non-async chroma clients interchangeably
async def maybe_await(result):
    if asyncio.iscoroutine(result):
        return await result
    return result


class ChromaIndex(EmbeddingIndex):
    def __init__(self, client: ChromaClientType, collection):
        self.client = client
        self.collection = collection

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(
            embeddings
        ), f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"

        await maybe_await(
            self.collection.add(
                documents=[chunk.model_dump_json() for chunk in chunks],
                embeddings=embeddings,
                ids=[f"{c.document_id}:chunk-{i}" for i, c in enumerate(chunks)],
            )
        )

    async def query(
        self, embedding: NDArray, k: int, score_threshold: float
    ) -> QueryDocumentsResponse:
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
        for dist, doc in zip(distances, documents):
            try:
                doc = json.loads(doc)
                chunk = Chunk(**doc)
            except Exception:
                log.exception(f"Failed to parse document: {doc}")
                continue

            chunks.append(chunk)
            scores.append(1.0 / float(dist))

        return QueryDocumentsResponse(chunks=chunks, scores=scores)

    async def delete(self):
        await maybe_await(self.client.delete_collection(self.collection.name))


class ChromaMemoryAdapter(Memory, MemoryBanksProtocolPrivate):
    def __init__(
        self,
        config: Union[ChromaRemoteImplConfig, ChromaInlineImplConfig],
        inference_api: Api.inference,
    ) -> None:
        log.info(f"Initializing ChromaMemoryAdapter with url: {config}")
        self.config = config
        self.inference_api = inference_api

        self.client = None
        self.cache = {}

    async def initialize(self) -> None:
        if isinstance(self.config, ChromaRemoteImplConfig):
            log.info(f"Connecting to Chroma server at: {self.config.url}")
            url = self.config.url.rstrip("/")
            parsed = urlparse(url)

            if parsed.path and parsed.path != "/":
                raise ValueError("URL should not contain a path")

            self.client = await chromadb.AsyncHttpClient(
                host=parsed.hostname, port=parsed.port
            )
        else:
            log.info(f"Connecting to Chroma local db at: {self.config.db_path}")
            self.client = chromadb.PersistentClient(path=self.config.db_path)

    async def shutdown(self) -> None:
        pass

    async def register_memory_bank(
        self,
        memory_bank: MemoryBank,
    ) -> None:
        assert (
            memory_bank.memory_bank_type == MemoryBankType.vector.value
        ), f"Only vector banks are supported {memory_bank.memory_bank_type}"

        collection = await maybe_await(
            self.client.get_or_create_collection(
                name=memory_bank.identifier,
                metadata={"bank": memory_bank.model_dump_json()},
            )
        )
        self.cache[memory_bank.identifier] = BankWithIndex(
            memory_bank, ChromaIndex(self.client, collection), self.inference_api
        )

    async def unregister_memory_bank(self, memory_bank_id: str) -> None:
        await self.cache[memory_bank_id].index.delete()
        del self.cache[memory_bank_id]

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        index = await self._get_and_cache_bank_index(bank_id)

        await index.insert_documents(documents)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedContent,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        index = await self._get_and_cache_bank_index(bank_id)

        return await index.query_documents(query, params)

    async def _get_and_cache_bank_index(self, bank_id: str) -> BankWithIndex:
        if bank_id in self.cache:
            return self.cache[bank_id]

        bank = await self.memory_bank_store.get_memory_bank(bank_id)
        if not bank:
            raise ValueError(f"Bank {bank_id} not found in Llama Stack")
        collection = await maybe_await(self.client.get_collection(bank_id))
        if not collection:
            raise ValueError(f"Bank {bank_id} not found in Chroma")
        index = BankWithIndex(
            bank, ChromaIndex(self.client, collection), self.inference_api
        )
        self.cache[bank_id] = index
        return index
