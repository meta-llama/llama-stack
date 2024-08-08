# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel


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
