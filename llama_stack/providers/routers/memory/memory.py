# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Tuple

from llama_stack.distribution.datatypes import Api
from llama_stack.apis.memory import *  # noqa: F403


class MemoryRouterImpl(Memory):
    """Routes to an provider based on the memory bank type"""

    def __init__(
        self,
        inner_impls: List[Tuple[str, Any]],
        deps: List[Api],
    ) -> None:
        self.deps = deps

        bank_types = [v.value for v in MemoryBankType]

        self.providers = {}
        for routing_key, provider_impl in inner_impls:
            if routing_key not in bank_types:
                raise ValueError(
                    f"Unknown routing key `{routing_key}` for memory bank type"
                )
            self.providers[routing_key] = provider_impl

        self.bank_id_to_type = {}

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        for p in self.providers.values():
            await p.shutdown()

    def get_provider(self, bank_type):
        if bank_type not in self.providers:
            raise ValueError(f"Memory bank type {bank_type} not supported")

        return self.providers[bank_type]

    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank:
        provider = self.get_provider(config.type)
        bank = await provider.create_memory_bank(name, config, url)
        self.bank_id_to_type[bank.bank_id] = config.type
        return bank

    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]:
        bank_type = self.bank_id_to_type.get(bank_id)
        if not bank_type:
            raise ValueError(f"Could not find bank type for {bank_id}")

        provider = self.get_provider(bank_type)
        return await provider.get_memory_bank(bank_id)

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        bank_type = self.bank_id_to_type.get(bank_id)
        if not bank_type:
            raise ValueError(f"Could not find bank type for {bank_id}")

        provider = self.get_provider(bank_type)
        return await provider.insert_documents(bank_id, documents, ttl_seconds)

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        bank_type = self.bank_id_to_type.get(bank_id)
        if not bank_type:
            raise ValueError(f"Could not find bank type for {bank_id}")

        provider = self.get_provider(bank_type)
        return await provider.query_documents(bank_id, query, params)
