# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field


@json_schema_type
class ShieldType(Enum):
    generic_content_shield = "generic_content_shield"
    llama_guard = "llama_guard"
    code_scanner = "code_scanner"
    prompt_guard = "prompt_guard"


class ShieldDef(BaseModel):
    identifier: str = Field(
        description="A unique identifier for the shield type",
    )
    shield_type: str = Field(
        description="The type of shield this is; the value is one of the ShieldType enum"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional parameters needed for this shield",
    )


@json_schema_type
class ShieldDefWithProvider(ShieldDef):
    type: Literal["shield"] = "shield"
    provider_id: str = Field(
        description="The provider ID for this shield type",
    )


@runtime_checkable
class Shields(Protocol):
    @webmethod(route="/shields/list", method="GET")
    async def list_shields(self) -> List[ShieldDefWithProvider]: ...

    @webmethod(route="/shields/get", method="GET")
    async def get_shield(self, shield_type: str) -> Optional[ShieldDefWithProvider]: ...

    @webmethod(route="/shields/register", method="POST")
    async def register_shield(self, shield: ShieldDefWithProvider) -> None: ...
