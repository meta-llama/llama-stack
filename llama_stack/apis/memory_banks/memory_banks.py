# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import List, Literal, Optional, Protocol, runtime_checkable, Union

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field
from typing_extensions import Annotated


@json_schema_type
class MemoryBankType(Enum):
    vector = "vector"
    keyvalue = "keyvalue"
    keyword = "keyword"
    graph = "graph"


class CommonDef(BaseModel):
    identifier: str
    # Hack: move this out later
    provider_id: str = ""


@json_schema_type
class VectorMemoryBankDef(CommonDef):
    type: Literal[MemoryBankType.vector.value] = MemoryBankType.vector.value
    embedding_model: str
    chunk_size_in_tokens: int
    overlap_size_in_tokens: Optional[int] = None


@json_schema_type
class KeyValueMemoryBankDef(CommonDef):
    type: Literal[MemoryBankType.keyvalue.value] = MemoryBankType.keyvalue.value


@json_schema_type
class KeywordMemoryBankDef(CommonDef):
    type: Literal[MemoryBankType.keyword.value] = MemoryBankType.keyword.value


@json_schema_type
class GraphMemoryBankDef(CommonDef):
    type: Literal[MemoryBankType.graph.value] = MemoryBankType.graph.value


MemoryBankDef = Annotated[
    Union[
        VectorMemoryBankDef,
        KeyValueMemoryBankDef,
        KeywordMemoryBankDef,
        GraphMemoryBankDef,
    ],
    Field(discriminator="type"),
]

MemoryBankDefWithProvider = MemoryBankDef


@runtime_checkable
class MemoryBanks(Protocol):
    @webmethod(route="/memory_banks/list", method="GET")
    async def list_memory_banks(self) -> List[MemoryBankDefWithProvider]: ...

    @webmethod(route="/memory_banks/get", method="GET")
    async def get_memory_bank(
        self, identifier: str
    ) -> Optional[MemoryBankDefWithProvider]: ...

    @webmethod(route="/memory_banks/register", method="POST")
    async def register_memory_bank(
        self, memory_bank: MemoryBankDefWithProvider
    ) -> None: ...
