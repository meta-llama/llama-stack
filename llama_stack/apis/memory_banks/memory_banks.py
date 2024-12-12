# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import (
    Annotated,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
    Union,
)

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel, Field

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol


@json_schema_type
class MemoryBankType(Enum):
    vector = "vector"
    keyvalue = "keyvalue"
    keyword = "keyword"
    graph = "graph"


# define params for each type of memory bank, this leads to a tagged union
# accepted as input from the API or from the config.
@json_schema_type
class VectorMemoryBankParams(BaseModel):
    memory_bank_type: Literal[MemoryBankType.vector.value] = MemoryBankType.vector.value
    embedding_model: str
    chunk_size_in_tokens: int
    overlap_size_in_tokens: Optional[int] = None


@json_schema_type
class KeyValueMemoryBankParams(BaseModel):
    memory_bank_type: Literal[MemoryBankType.keyvalue.value] = (
        MemoryBankType.keyvalue.value
    )


@json_schema_type
class KeywordMemoryBankParams(BaseModel):
    memory_bank_type: Literal[MemoryBankType.keyword.value] = (
        MemoryBankType.keyword.value
    )


@json_schema_type
class GraphMemoryBankParams(BaseModel):
    memory_bank_type: Literal[MemoryBankType.graph.value] = MemoryBankType.graph.value


BankParams = Annotated[
    Union[
        VectorMemoryBankParams,
        KeyValueMemoryBankParams,
        KeywordMemoryBankParams,
        GraphMemoryBankParams,
    ],
    Field(discriminator="memory_bank_type"),
]


# Some common functionality for memory banks.
class MemoryBankResourceMixin(Resource):
    type: Literal[ResourceType.memory_bank.value] = ResourceType.memory_bank.value

    @property
    def memory_bank_id(self) -> str:
        return self.identifier

    @property
    def provider_memory_bank_id(self) -> str:
        return self.provider_resource_id


@json_schema_type
class VectorMemoryBank(MemoryBankResourceMixin):
    memory_bank_type: Literal[MemoryBankType.vector.value] = MemoryBankType.vector.value
    embedding_model: str
    chunk_size_in_tokens: int
    overlap_size_in_tokens: Optional[int] = None


@json_schema_type
class KeyValueMemoryBank(MemoryBankResourceMixin):
    memory_bank_type: Literal[MemoryBankType.keyvalue.value] = (
        MemoryBankType.keyvalue.value
    )


# TODO: KeyValue and Keyword are so similar in name, oof. Get a better naming convention.
@json_schema_type
class KeywordMemoryBank(MemoryBankResourceMixin):
    memory_bank_type: Literal[MemoryBankType.keyword.value] = (
        MemoryBankType.keyword.value
    )


@json_schema_type
class GraphMemoryBank(MemoryBankResourceMixin):
    memory_bank_type: Literal[MemoryBankType.graph.value] = MemoryBankType.graph.value


MemoryBank = Annotated[
    Union[
        VectorMemoryBank,
        KeyValueMemoryBank,
        KeywordMemoryBank,
        GraphMemoryBank,
    ],
    Field(discriminator="memory_bank_type"),
]


class MemoryBankInput(BaseModel):
    memory_bank_id: str
    params: BankParams
    provider_memory_bank_id: Optional[str] = None


@runtime_checkable
@trace_protocol
class MemoryBanks(Protocol):
    @webmethod(route="/memory-banks/list", method="GET")
    async def list_memory_banks(self) -> List[MemoryBank]: ...

    @webmethod(route="/memory-banks/get", method="GET")
    async def get_memory_bank(self, memory_bank_id: str) -> Optional[MemoryBank]: ...

    @webmethod(route="/memory-banks/register", method="POST")
    async def register_memory_bank(
        self,
        memory_bank_id: str,
        params: BankParams,
        provider_id: Optional[str] = None,
        provider_memory_bank_id: Optional[str] = None,
    ) -> MemoryBank: ...

    @webmethod(route="/memory-banks/unregister", method="POST")
    async def unregister_memory_bank(self, memory_bank_id: str) -> None: ...
