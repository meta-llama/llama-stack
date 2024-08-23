# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Protocol

from llama_models.llama3.api.datatypes import InterleavedTextMedia

from llama_models.schema_utils import webmethod
from .datatypes import *  # noqa: F403


@json_schema_type
class RetrieveMemoryDocumentsRequest(BaseModel):
    query: InterleavedTextMedia
    bank_ids: str


@json_schema_type
class RetrieveMemoryDocumentsResponse(BaseModel):
    documents: List[MemoryBankDocument]
    scores: List[float]


class Memory(Protocol):
    @webmethod(route="/memory_banks/create")
    def create_memory_bank(
        self,
        bank_id: str,
        bank_name: str,
        embedding_model: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_banks/list")
    def get_memory_banks(self) -> List[MemoryBank]: ...

    @webmethod(route="/memory_banks/get")
    def get_memory_bank(self, bank_id: str) -> List[MemoryBank]: ...

    @webmethod(route="/memory_banks/drop")
    def delete_memory_bank(
        self,
        bank_id: str,
    ) -> str: ...

    @webmethod(route="/memory_bank/insert")
    def insert_memory_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_bank/update")
    def update_memory_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_bank/get")
    def retrieve_memory_documents(
        self,
        request: RetrieveMemoryDocumentsRequest,
    ) -> List[MemoryBankDocument]: ...

    @webmethod(route="/memory_bank/get")
    def get_memory_documents(
        self,
        bank_id: str,
        document_uuids: List[str],
    ) -> List[MemoryBankDocument]: ...

    @webmethod(route="/memory_bank/delete")
    def delete_memory_documents(
        self,
        bank_id: str,
        document_uuids: List[str],
    ) -> List[str]: ...
