# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field

from llama_stack.apis.memory import MemoryBankType

from llama_stack.distribution.datatypes import GenericProviderConfig


@json_schema_type
class MemoryBankSpec(BaseModel):
    bank_type: MemoryBankType
    provider_config: GenericProviderConfig = Field(
        description="Provider config for the model, including provider_type, and corresponding config. ",
    )


class MemoryBanks(Protocol):
    @webmethod(route="/memory_banks/list", method="GET")
    async def list_available_memory_banks(self) -> List[MemoryBankSpec]: ...

    @webmethod(route="/memory_banks/get", method="GET")
    async def get_serving_memory_bank(
        self, bank_type: MemoryBankType
    ) -> Optional[MemoryBankSpec]: ...
