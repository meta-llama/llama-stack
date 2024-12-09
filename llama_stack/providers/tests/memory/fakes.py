# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

from llama_stack.apis.memory.memory import (
    Memory,
    MemoryBankDocument,
    MemoryBankStore,
    QueryDocumentsResponse,
)
from llama_stack.apis.memory_banks.memory_banks import KeyValueMemoryBank
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403


# MemoryBanks test fake implementation to
# support behaviors tested in test_memory.py
class MemoryBanksTestFakeImpl(MemoryBanks):
    def __init__(self):
        self.memory_banks: Dict[str, MemoryBank] = dict()

    async def list_memory_banks(self) -> List[MemoryBank]:
        return list(self.memory_banks.values())

    async def get_memory_bank(self, memory_bank_id: str) -> Optional[MemoryBank]:
        if memory_bank_id in self.memory_banks:
            return self.memory_banks[memory_bank_id]

    async def register_memory_bank(
        self,
        memory_bank_id: str,
        params: BankParams,
        provider_id: Optional[str] = None,
        provider_memory_bank_id: Optional[str] = None,
    ) -> MemoryBank:
        memory_bank = KeyValueMemoryBank(
            identifier=memory_bank_id,
            provider_id="test::test-fake",
        )
        self.memory_banks[memory_bank_id] = memory_bank
        return self.memory_banks[memory_bank_id]

    async def unregister_memory_bank(self, memory_bank_id: str) -> None:
        if memory_bank_id in self.memory_banks:
            del self.memory_banks[memory_bank_id]


# Memory test fake implementation to
# support behaviors tested in test_memory.py
class MemoryTestFakeImpl(Memory):
    memory_bank_store: MemoryBankStore

    def __init__(self):
        self.memory_banks = None
        self.stubs: Dict[str, Any] = {}

    def set_memory_banks(self, memory_banks: MemoryBanks) -> None:
        self.memory_banks = memory_banks

    def set_stubs(self, method: str, stubs: Dict[str, Any]):
        self.stubs[method] = stubs

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        if not await self.memory_banks.get_memory_bank(bank_id):
            raise ValueError(f"Bank {bank_id} not found")
        # No-op
        # We will just ignore documents here since we will init this
        # test fake with stubs to match expecting query-response pairs

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        if not await self.memory_banks.get_memory_bank(bank_id):
            raise ValueError(f"Bank {bank_id} not found")
        if query not in self.stubs["query_documents"]:
            raise ValueError(
                f"Stub not created for query {query}, please check your test setup."
            )

        return self.stubs["query_documents"][query]
