# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, HttpUrl, field_validator
from pydantic_core import PydanticCustomError

from llama_stack.providers.datatypes import HealthResponse
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class ProviderInfo(BaseModel):
    api: str
    provider_id: str
    provider_type: str
    config: dict[str, Any]
    health: HealthResponse
    metrics: str | None = Field(
        default=None, description="Endpoint for metrics from providers. Must be a valid HTTP URL if provided."
    )

    @field_validator("metrics")
    @classmethod
    def validate_metrics_url(cls, v):
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("'metrics' must be a string URL or None")
        try:
            HttpUrl(v)  # Validate the URL
            return v
        except (PydanticCustomError, ValueError) as e:
            raise ValueError(f"'metrics' must be a valid HTTP or HTTPS URL: {str(e)}") from e


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
