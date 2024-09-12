# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_models.llama3.api.datatypes import *  # noqa: F403


@json_schema_type
class MemoryBankDocument(BaseModel):
    document_id: str
    content: InterleavedTextMedia | URL
    mime_type: str | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


@json_schema_type
class MemoryBankType(Enum):
    vector = "vector"
    keyvalue = "keyvalue"
    keyword = "keyword"
    graph = "graph"


class VectorMemoryBankConfig(BaseModel):
    type: Literal[MemoryBankType.vector.value] = MemoryBankType.vector.value
    embedding_model: str
    chunk_size_in_tokens: int
    overlap_size_in_tokens: Optional[int] = None


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


class Chunk(BaseModel):
    content: InterleavedTextMedia
    token_count: int
    document_id: str


@json_schema_type
class QueryDocumentsResponse(BaseModel):
    chunks: List[Chunk]
    scores: List[float]


@json_schema_type
class QueryAPI(Protocol):
    @webmethod(route="/query_documents")
    def query_documents(
        self,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse: ...


@json_schema_type
class MemoryBank(BaseModel):
    bank_id: str
    name: str
    config: MemoryBankConfig
    # if there's a pre-existing (reachable-from-distribution) store which supports QueryAPI
    url: Optional[URL] = None


class Memory(Protocol):
    @webmethod(route="/memory_banks/create")
    async def create_memory_bank(
        self,
        name: str,
        config: MemoryBankConfig,
        url: Optional[URL] = None,
    ) -> MemoryBank: ...

    @webmethod(route="/memory_banks/list", method="GET")
    async def list_memory_banks(self) -> List[MemoryBank]: ...

    @webmethod(route="/memory_banks/get", method="GET")
    async def get_memory_bank(self, bank_id: str) -> Optional[MemoryBank]: ...

    @webmethod(route="/memory_banks/drop", method="DELETE")
    async def drop_memory_bank(
        self,
        bank_id: str,
    ) -> str: ...

    # this will just block now until documents are inserted, but it should
    # probably return a Job instance which can be polled for completion
    @webmethod(route="/memory_bank/insert")
    async def insert_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
        ttl_seconds: Optional[int] = None,
    ) -> None: ...

    @webmethod(route="/memory_bank/update")
    async def update_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_bank/query")
    async def query_documents(
        self,
        bank_id: str,
        query: InterleavedTextMedia,
        params: Optional[Dict[str, Any]] = None,
    ) -> QueryDocumentsResponse: ...

    @webmethod(route="/memory_bank/documents/get", method="GET")
    async def get_documents(
        self,
        bank_id: str,
        document_ids: List[str],
    ) -> List[MemoryBankDocument]: ...

    @webmethod(route="/memory_bank/documents/delete", method="DELETE")
    async def delete_documents(
        self,
        bank_id: str,
        document_ids: List[str],
    ) -> None: ...
