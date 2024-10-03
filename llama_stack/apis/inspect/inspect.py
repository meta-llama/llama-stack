# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List, Protocol

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel


@json_schema_type
class ProviderInfo(BaseModel):
    provider_type: str
    description: str


@json_schema_type
class RouteInfo(BaseModel):
    route: str
    method: str
    providers: List[str]


@json_schema_type
class HealthInfo(BaseModel):
    status: str
    # TODO: add a provider level status


class Inspect(Protocol):
    @webmethod(route="/providers/list", method="GET")
    async def list_providers(self) -> Dict[str, ProviderInfo]: ...

    @webmethod(route="/routes/list", method="GET")
    async def list_routes(self) -> Dict[str, List[RouteInfo]]: ...

    @webmethod(route="/health", method="GET")
    async def health(self) -> HealthInfo: ...
