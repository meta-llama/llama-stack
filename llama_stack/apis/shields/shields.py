# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol


class CommonShieldFields(BaseModel):
    params: Optional[Dict[str, Any]] = None


@json_schema_type
class Shield(CommonShieldFields, Resource):
    """A safety shield resource that can be used to check content"""

    type: Literal[ResourceType.shield.value] = ResourceType.shield.value

    @property
    def shield_id(self) -> str:
        return self.identifier

    @property
    def provider_shield_id(self) -> str:
        return self.provider_resource_id


class ShieldInput(CommonShieldFields):
    shield_id: str
    provider_id: Optional[str] = None
    provider_shield_id: Optional[str] = None


@runtime_checkable
@trace_protocol
class Shields(Protocol):
    @webmethod(route="/shields/list", method="GET")
    async def list_shields(self) -> List[Shield]: ...

    @webmethod(route="/shields/get", method="GET")
    async def get_shield(self, identifier: str) -> Optional[Shield]: ...

    @webmethod(route="/shields/register", method="POST")
    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Shield: ...
