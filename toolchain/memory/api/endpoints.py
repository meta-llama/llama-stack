from typing import List, Protocol

from pyopenapi import webmethod

from .datatypes import *  # noqa: F403


class MemoryBanks(Protocol):
    @webmethod(route="/memory_banks/create")
    def post_create_memory_bank(
        self,
        bank_id: str,
        bank_name: str,
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
    def post_insert_memory_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_bank/update")
    def post_update_memory_documents(
        self,
        bank_id: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

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
