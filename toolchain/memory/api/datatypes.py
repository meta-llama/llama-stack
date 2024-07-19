from typing import Any, Dict

from pydantic import BaseModel

from strong_typing.schema import json_schema_type


@json_schema_type
class MemoryBank(BaseModel):
    memory_bank_id: str
    memory_bank_name: str


@json_schema_type
class MemoryBankDocument(BaseModel):
    document_id: str
    content: bytes
    metadata: Dict[str, Any]
    mime_type: str
