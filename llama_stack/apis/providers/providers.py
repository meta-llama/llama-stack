# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.providers.datatypes import HealthResponse
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ProviderInfo(BaseModel):
    api: str
    provider_id: str
    provider_type: str
    config: dict[str, Any]
    health: HealthResponse


class ListProvidersResponse(BaseModel):
    data: list[ProviderInfo]


@runtime_checkable
class Providers(Protocol):
    """
    Providers API for inspecting, listing, and modifying providers and their configurations.
    """

    @webmethod(route="/providers", method="GET")
    async def list_providers(self) -> ListProvidersResponse:
        """List all available providers.

        :returns: A ListProvidersResponse containing information about all providers.
        """
        ...

    @webmethod(route="/providers/{provider_id}", method="GET")
    async def inspect_provider(self, provider_id: str) -> ProviderInfo:
        """Get detailed information about a specific provider.

        :param provider_id: The ID of the provider to inspect.
        :returns: A ProviderInfo object containing the provider's details.
        """
        ...

    @webmethod(route="/providers/{api}/{provider_id}/{provider_type}", method="PUT")
    async def update_provider(
        self, api: str, provider_id: str, provider_type: str, config: dict[str, Any]
    ) -> ProviderInfo: ...
