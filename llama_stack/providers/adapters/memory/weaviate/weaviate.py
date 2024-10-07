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
from llama_stack.providers.utils.memory.vector_store import (
    BankWithIndex,
    EmbeddingIndex,
)

from .config import WeaviateConfig, WeaviateRequestProviderData


class WeaviateIndex(EmbeddingIndex):
    def __init__(self, client: weaviate.Client, collection: str):
        self.client = client
        self.collection = collection

    async def add_chunks(self, chunks: List[Chunk], embeddings: NDArray):
        assert len(chunks) == len(
            embeddings
        ), f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"

        data_objects = []
        for i, chunk in enumerate(chunks):
            data_objects.append(
                wvc.data.DataObject(
                    properties={
                        "chunk_content": chunk,
                    },
                    vector=embeddings[i].tolist(),
                )
            )

        # Inserting chunks into a prespecified Weaviate collection
        assert self.collection is not None, "Collection name must be specified"
        my_collection = self.client.collections.get(self.collection)

        await my_collection.data.insert_many(data_objects)

    async def query(self, embedding: NDArray, k: int) -> QueryDocumentsResponse:
        assert self.collection is not None, "Collection name must be specified"

        my_collection = self.client.collections.get(self.collection)

        results = my_collection.query.near_vector(
            near_vector=embedding.tolist(),
            limit=k,
            return_meta_data=wvc.query.MetadataQuery(distance=True),
        )

        chunks = []
        scores = []
        for doc in results.objects:
            try:
                chunk = doc.properties["chunk_content"]
                chunks.append(chunk)
                scores.append(1.0 / doc.metadata.distance)

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Failed to parse document: {e}")

        return QueryDocumentsResponse(chunks=chunks, scores=scores)


class WeaviateMemoryAdapter(Memory, NeedsRequestProviderData):
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

        client = await self._get_client()

        # Create collection if it doesn't exist
        if not client.collections.exists(memory_bank.identifier):
            client.collections.create(
                name=smemory_bank.identifier,
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
            index=WeaviateIndex(client=client, collection=memory_bank.identifier),
        )
        self.cache[bank_id] = index

    async def _get_and_cache_bank_index(self, bank_id: str) -> Optional[BankWithIndex]:
        if bank_id in self.cache:
            return self.cache[bank_id]

        bank = await self.memory_bank_store.get_memory_bank(bank_id)
        if not bank:
            raise ValueError(f"Bank {bank_id} not found")

        client = await self._get_client()
        collections = await client.collections.list_all().keys()

        for collection in collections:
            if collection == bank_id:
                bank = MemoryBank(**json.loads(collection.metadata["bank"]))
                index = BankWithIndex(
                    bank=bank,
                    index=WeaviateIndex(self.client, collection),
                )
                self.cache[bank_id] = index
                return index

        return None

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
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
