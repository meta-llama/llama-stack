# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Protocol

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_models.schema_utils import webmethod
from .datatypes import *  # noqa: F403


@json_schema_type
class MemoryBankDocument(BaseModel):
    document_id: str
    content: InterleavedTextMedia | URL
    mime_type: str
    metadata: Dict[str, Any]


class Chunk(BaseModel):
    content: InterleavedTextMedia
    token_count: int


@json_schema_type
class QueryDocumentsResponse(BaseModel):
    chunks: List[Chunk]
    scores: List[float]


@json_schema_type
class MemoryBankType(Enum):
    vector = "vector"
    keyvalue = "keyvalue"
    keyword = "keyword"
    graph = "graph"


class VectorMemoryBankConfig(BaseModel):
    type: Literal[MemoryBankType.vector.value] = MemoryBankType.vector.value
    embedding_model: str


class KeyValueMemoryBankConfig(BaseModel):
    type: Literal[MemoryBankType.keyvalue.value] = MemoryBankType.keyvalue.value


class KeywordMemoryBankConfig(BaseModel):
    type: Literal[MemoryBankType.keyword.value] = MemoryBankType.keyword.value


class GraphMemoryBankConfig(BaseModel):
    type: Literal[MemoryBankType.graph.value] = MemoryBankType.graph.value


MemoryBankConfig = Annotated[
    Union[
        VectorMemoryBankConfig,
        KeyValueMemoryBankConfig,
        KeywordMemoryBankConfig,
        GraphMemoryBankConfig,
    ],
    Field(discriminator="type"),
]


@json_schema_type
class MemoryBank(BaseModel):
    bank_id: str
    name: str
    config: MemoryBankConfig
    # if there's a pre-existing store which obeys the MemoryBank REST interface
    url: Optional[URL] = None


class Memory(Protocol):
    @webmethod(route="/memory_banks/create")
    def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank: ...

    @webmethod(route="/memory_banks/list", method="GET")
    def list_memory_banks(self) -> List[MemoryBank]: ...

    @webmethod(route="/memory_banks/get")
    def get_memory_bank(self, bank_id: str) -> MemoryBank: ...

    @webmethod(route="/memory_banks/drop", method="DELETE")
    def drop_memory_bank(
        self,
        bank_id: str,
    ) -> str: ...

    @webmethod(route="/memory_bank/insert")
    def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_bank/update")
    def update_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_bank/query")
    def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse: ...

    @webmethod(route="/memory_bank/documents/get")
    def get_documents(
        self,
        bank_id: str,
        document_ids: List[str],
    ) -> List[MemoryBankDocument]: ...

    @webmethod(route="/memory_bank/documents/delete")
    def delete_documents(
        self,
        bank_id: str,
        document_ids: List[str],
    ) -> None: ...
