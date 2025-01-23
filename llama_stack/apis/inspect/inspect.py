# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel


@json_schema_type
class ProviderInfo(BaseModel):
    api: str
    provider_id: str
    provider_type: str


@json_schema_type
class RouteInfo(BaseModel):
    route: str
    method: str
    provider_types: List[str]


@json_schema_type
class HealthInfo(BaseModel):
    status: str
    # TODO: add a provider level status


@json_schema_type
class VersionInfo(BaseModel):
    version: str


class ListProvidersResponse(BaseModel):
    data: List[ProviderInfo]


class ListRoutesResponse(BaseModel):
    data: List[RouteInfo]


@runtime_checkable
class Inspect(Protocol):
    @webmethod(route="/inspect/providers", method="GET")
    async def list_providers(self) -> ListProvidersResponse: ...

    @webmethod(route="/inspect/routes", method="GET")
    async def list_routes(self) -> ListRoutesResponse: ...

    @webmethod(route="/health", method="GET")
    async def health(self) -> HealthInfo: ...

    @webmethod(route="/version", method="GET")
    async def version(self) -> VersionInfo: ...
