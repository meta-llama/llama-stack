# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Annotated, List, Literal, Union

from llama_stack.distribution.utils.config_dirs import RUNTIME_BASE_DIR
from llama_stack.providers.utils.kvstore import KVStoreConfig, SqliteKVStoreConfig

from pydantic import BaseModel, Field


class _MemoryBankConfigCommon(BaseModel):
    bank_id: str


class VectorMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal["vector"] = "vector"


class KeyValueMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal["keyvalue"] = "keyvalue"
    keys: List[str]  # what keys to focus on


class KeywordMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal["keyword"] = "keyword"


class GraphMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal["graph"] = "graph"
    entities: List[str]  # what entities to focus on


MemoryBankConfig = Annotated[
    Union[
        VectorMemoryBankConfig,
        KeyValueMemoryBankConfig,
        KeywordMemoryBankConfig,
        GraphMemoryBankConfig,
    ],
    Field(discriminator="type"),
]


class MemoryQueryGenerator(Enum):
    default = "default"
    llm = "llm"
    custom = "custom"


class DefaultMemoryQueryGeneratorConfig(BaseModel):
    type: Literal[MemoryQueryGenerator.default.value] = (
        MemoryQueryGenerator.default.value
    )
    sep: str = " "


class LLMMemoryQueryGeneratorConfig(BaseModel):
    type: Literal[MemoryQueryGenerator.llm.value] = MemoryQueryGenerator.llm.value
    model: str
    template: str


class CustomMemoryQueryGeneratorConfig(BaseModel):
    type: Literal[MemoryQueryGenerator.custom.value] = MemoryQueryGenerator.custom.value


MemoryQueryGeneratorConfig = Annotated[
    Union[
        DefaultMemoryQueryGeneratorConfig,
        LLMMemoryQueryGeneratorConfig,
        CustomMemoryQueryGeneratorConfig,
    ],
    Field(discriminator="type"),
]


class MemoryToolConfig(BaseModel):
    memory_bank_configs: List[MemoryBankConfig] = Field(default_factory=list)
    # This config defines how a query is generated using the messages
    # for memory bank retrieval.
    query_generator_config: MemoryQueryGeneratorConfig = Field(
        default=DefaultMemoryQueryGeneratorConfig()
    )
    max_tokens_in_context: int = 4096
    max_chunks: int = 10
    kvstore_config: KVStoreConfig = SqliteKVStoreConfig(
        db_path=(RUNTIME_BASE_DIR / "memory.db").as_posix()
    )
