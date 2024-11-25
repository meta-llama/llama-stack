# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.memory_banks import MemoryBank
from llama_stack.providers.datatypes import MemoryBanksProtocolPrivate


@json_schema_type
class InlineMemoryFakeConfig(BaseModel):
    pass


class InlineMemoryFakeImpl(Memory, MemoryBanksProtocolPrivate):
    method_stubs: Dict[str, Any] = {}
    memory_banks: Dict[str, MemoryBank] = {}

    @staticmethod
    def stub_method(method_name: str, return_value_matchers: Dict[str, Any]) -> None:
        if method_name in InlineMemoryFakeImpl.method_stubs:
            InlineMemoryFakeImpl.method_stubs[method_name].update(return_value_matchers)
            return
        InlineMemoryFakeImpl.method_stubs[method_name] = return_value_matchers

    def __init__(self, config: InlineMemoryFakeConfig) -> None:
        pass

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_memory_bank(
        self,
        memory_bank: MemoryBank,
    ) -> None:
        InlineMemoryFakeImpl.memory_banks[memory_bank.memory_bank_id] = memory_bank

    async def list_memory_banks(self) -> List[MemoryBank]:
        return list(InlineMemoryFakeImpl.memory_banks.values())

    async def unregister_memory_bank(self, memory_bank_id: str) -> None:
        if memory_bank_id not in InlineMemoryFakeImpl.memory_banks:
            raise ValueError(f"Bank {memory_bank_id} not found.")
        del InlineMemoryFakeImpl.memory_banks[memory_bank_id]

    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None:
        pass

    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse:
        if query in InlineMemoryFakeImpl.method_stubs["query_documents"]:
            return InlineMemoryFakeImpl.method_stubs["query_documents"][query]
        raise ValueError(
            f"Stub for query '{query}' not found, please set up expected result"
        )


async def get_provider_impl(config: InlineMemoryFakeConfig, _deps: Any):
    assert isinstance(
        config, InlineMemoryFakeConfig
    ), f"Unexpected config type: {type(config)}"

    impl = InlineMemoryFakeImpl(config)
    await impl.initialize()
    return impl
