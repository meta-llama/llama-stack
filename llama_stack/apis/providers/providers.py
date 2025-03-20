# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ProviderInfo(BaseModel):
    api: str
    provider_id: str
    provider_type: str
    config: Dict[str, Any]


class ListProvidersResponse(BaseModel):
    data: List[ProviderInfo]


@runtime_checkable
class Providers(Protocol):
    """
    Providers API for inspecting, listing, and modifying providers and their configurations.
    """

    @webmethod(route="/providers", method="GET")
    async def list_providers(self) -> ListProvidersResponse: ...

    @webmethod(route="/providers/{provider_id}", method="GET")
    async def inspect_provider(self, provider_id: str) -> ProviderInfo: ...
