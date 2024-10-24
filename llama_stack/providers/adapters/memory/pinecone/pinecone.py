# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time

from numpy.typing import NDArray
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import MemoryBanksProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import (
    BankWithIndex,
    EmbeddingIndex,
)
from .config import PineconeConfig, PineconeRequestProviderData


class PineconeIndex(EmbeddingIndex):
    def __init__(self, client: Pinecone, index_name: str):
        self.client = client
        self.index_name = index_name

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(
            embeddings
        ), f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"

        data_objects = []
        for i, chunk in enumerate(chunks):
            data_objects.append(
                {
                    "id": chunk.document_id,
                    "values": embeddings[i].tolist(),
                    "metadata": {
                        "content": chunk.content,
                        "token_count": chunk.token_count,
                        "document_id": chunk.document_id,
                    },
                }
            )

        # Inserting chunks into a prespecified Weaviate collection
        index = self.client.Index(self.index_name)
        index.upsert(vectors=data_objects)
        time.sleep(1)

    async def query(
        self, embedding: NDArray, k: int, score_threshold: float
    ) -> QueryDocumentsResponse:
        index = self.client.Index(self.index_name)

        results = index.query(
            vector=embedding, top_k=k, include_values=False, include_metadata=True
        )

        chunks = []
        scores = []
        for doc in results["matches"]:
            chunk_json = doc["metadata"]
            try:
                chunk = Chunk(**chunk_json)
            except Exception:
                import traceback

                traceback.print_exc()
                print(f"Failed to parse document: {chunk_json}")
                continue

            chunks.append(chunk)
            scores.append(doc.score)

        return QueryDocumentsResponse(chunks=chunks, scores=scores)


class PineconeMemoryAdapter(
    Memory, NeedsRequestProviderData, MemoryBanksProtocolPrivate
):
    def __init__(self, config: PineconeConfig) -> None:
        self.config = config
        self.client_cache = {}
        self.cache = {}

    def _get_client(self) -> Pinecone:
        provider_data = self.get_request_provider_data()
        assert provider_data is not None, "Request provider data must be set"
        assert isinstance(provider_data, PineconeRequestProviderData)

        key = f"{provider_data.pinecone_api_key}"
        if key in self.client_cache:
            return self.client_cache[key]

        client = Pinecone(api_key=provider_data.pinecone_api_key)
        self.client_cache[key] = client
        return client

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def check_if_index_exists(
        self,
        client: Pinecone,
        index_name: str,
    ) -> bool:
        try:
            # Get list of all indexes
            active_indexes = client.list_indexes()
            for index in active_indexes:
                if index["name"] == index_name:
                    return True
            return False
        except Exception as e:
            print(f"Error checking index: {e}")
            return False

    async def register_memory_bank(
        self,
        memory_bank: MemoryBankDef,
    ) -> None:
        assert (
            memory_bank.type == MemoryBankType.vector.value
        ), f"Only vector banks are supported {memory_bank.type}"

        client = self._get_client()

        # Create collection if it doesn't exist
        if not self.check_if_index_exists(client, memory_bank.identifier):
            client.create_index(
                name=memory_bank.identifier,
                dimension=self.config.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.config.cloud,
                    region=self.config.region,
                ),
            )

        index = BankWithIndex(
            bank=memory_bank,
            index=PineconeIndex(client=client, index_name=memory_bank.identifier),
        )
        self.cache[memory_bank.identifier] = index

    async def list_memory_banks(self) -> List[MemoryBankDef]:
        # TODO: right now the Llama Stack is the source of truth for these banks. That is
        # not ideal. It should be pinecone which is the source of truth. Unfortunately,
        # list() happens at Stack startup when the Pinecone client (credentials) is not
        # yet available. We need to figure out a way to make this work.
        return [i.bank for i in self.cache.values()]

    async def _get_and_cache_bank_index(self, bank_id: str) -> Optional[BankWithIndex]:
        if bank_id in self.cache:
            return self.cache[bank_id]

        bank = await self.memory_bank_store.get_memory_bank(bank_id)
        if not bank:
            raise ValueError(f"Bank {bank_id} not found")

        client = self._get_client()
        if not self.check_if_index_exists(client, bank_id):
            raise ValueError(f"Collection with name `{bank_id}` not found")

        index = BankWithIndex(
            bank=bank,
            index=PineconeIndex(client=client, index_name=bank_id),
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
