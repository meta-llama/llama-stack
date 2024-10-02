# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field

from llama_stack.distribution.datatypes import GenericProviderConfig


@json_schema_type
class ShieldSpec(BaseModel):
    shield_type: str
    provider_config: GenericProviderConfig = Field(
        description="Provider config for the model, including provider_type, and corresponding config. ",
    )


class Shields(Protocol):
    @webmethod(route="/shields/list", method="GET")
    async def list_shields(self) -> List[ShieldSpec]: ...

    @webmethod(route="/shields/get", method="GET")
    async def get_shield(self, shield_type: str) -> Optional[ShieldSpec]: ...
