# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json

from typing import Any, Dict, List, Optional

import weaviate
import weaviate.classes as wvc
from numpy.typing import NDArray
from weaviate.classes.init import Auth

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import MemoryBanksProtocolPrivate
from llama_stack.providers.utils.memory.vector_store import (
    BankWithIndex,
    EmbeddingIndex,
)

from .config import WeaviateConfig, WeaviateRequestProviderData


class WeaviateIndex(EmbeddingIndex):
    def __init__(self, client: weaviate.Client, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(
            embeddings
        ), f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"

        data_objects = []
        for i, chunk in enumerate(chunks):
            data_objects.append(
                wvc.data.DataObject(
                    properties={
                        "chunk_content": chunk.json(),
                    },
                    vector=embeddings[i].tolist(),
                )
            )

        # Inserting chunks into a prespecified Weaviate collection
        collection = self.client.collections.get(self.collection_name)

        # TODO: make this async friendly
        collection.data.insert_many(data_objects)

    async def query(
        self, embedding: NDArray, k: int, score_threshold: float
    ) -> QueryDocumentsResponse:
        collection = self.client.collections.get(self.collection_name)

        results = collection.query.near_vector(
            near_vector=embedding.tolist(),
            limit=k,
            return_metadata=wvc.query.MetadataQuery(distance=True),
        )

        chunks = []
        scores = []
        for doc in results.objects:
            chunk_json = doc.properties["chunk_content"]
            try:
                chunk_dict = json.loads(chunk_json)
                chunk = Chunk(**chunk_dict)
            except Exception:
                import traceback

                traceback.print_exc()
                print(f"Failed to parse document: {chunk_json}")
                continue

            chunks.append(chunk)
            scores.append(1.0 / doc.metadata.distance)

        return QueryDocumentsResponse(chunks=chunks, scores=scores)


class WeaviateMemoryAdapter(
    Memory, NeedsRequestProviderData, MemoryBanksProtocolPrivate
):
    def __init__(self, config: WeaviateConfig) -> None:
        self.config = config
        self.client_cache = {}
        self.cache = {}

    def _get_client(self) -> weaviate.Client:
        provider_data = self.get_request_provider_data()
        assert provider_data is not None, "Request provider data must be set"
        assert isinstance(provider_data, WeaviateRequestProviderData)

        key = f"{provider_data.weaviate_cluster_url}::{provider_data.weaviate_api_key}"
        if key in self.client_cache:
            return self.client_cache[key]

        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=provider_data.weaviate_cluster_url,
            auth_credentials=Auth.api_key(provider_data.weaviate_api_key),
        )
        self.client_cache[key] = client
        return client

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        for client in self.client_cache.values():
            client.close()

    async def register_memory_bank(
        self,
        memory_bank: MemoryBankDef,
    ) -> None:
        assert (
            memory_bank.type == MemoryBankType.vector.value
        ), f"Only vector banks are supported {memory_bank.type}"

        client = self._get_client()

        # Create collection if it doesn't exist
        if not client.collections.exists(memory_bank.identifier):
            client.collections.create(
                name=memory_bank.identifier,
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
                properties=[
                    wvc.config.Property(
                        name="chunk_content",
                        data_type=wvc.config.DataType.TEXT,
                    ),
                ],
            )

        index = BankWithIndex(
            bank=memory_bank,
            index=WeaviateIndex(client=client, collection_name=memory_bank.identifier),
        )
        self.cache[memory_bank.identifier] = index

    async def list_memory_banks(self) -> List[MemoryBankDef]:
        # TODO: right now the Llama Stack is the source of truth for these banks. That is
        # not ideal. It should be Weaviate which is the source of truth. Unfortunately,
        # list() happens at Stack startup when the Weaviate client (credentials) is not
        # yet available. We need to figure out a way to make this work.
        return [i.bank for i in self.cache.values()]

    async def _get_and_cache_bank_index(self, bank_id: str) -> Optional[BankWithIndex]:
        if bank_id in self.cache:
            return self.cache[bank_id]

        bank = await self.memory_bank_store.get_memory_bank(bank_id)
        if not bank:
            raise ValueError(f"Bank {bank_id} not found")

        client = self._get_client()
        if not client.collections.exists(bank_id):
            raise ValueError(f"Collection with name `{bank_id}` not found")

        index = BankWithIndex(
            bank=bank,
            index=WeaviateIndex(client=client, collection_name=bank_id),
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
