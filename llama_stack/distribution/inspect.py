# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from importlib.metadata import version
from typing import Any, Dict

from pydantic import BaseModel

from llama_stack.apis.inspect import (
    HealthInfo,
    Inspect,
    ListProvidersResponse,
    ListRoutesResponse,
    ProviderInfo,
    RouteInfo,
    VersionInfo,
)
from llama_stack.distribution.datatypes import StackRunConfig
from llama_stack.distribution.server.endpoints import get_all_api_endpoints
from llama_stack.providers.datatypes import Api, HealthResponse, HealthStatus


class DistributionInspectConfig(BaseModel):
    run_config: StackRunConfig


async def get_provider_impl(config, deps):
    impl = DistributionInspectImpl(config, deps)
    await impl.initialize()
    return impl


class DistributionInspectImpl(Inspect):
    def __init__(self, config, deps):
        self.config = config
        self.deps = deps
        self.impls = Dict[Api, Any]  # list of providers implementations

    async def initialize(self) -> None:
        pass

    async def list_providers(self) -> ListProvidersResponse:
        run_config: StackRunConfig = self.config.run_config
        providers_health = await self.get_providers_health()
        ret = []
        for api, providers in run_config.providers.items():
            for p in providers:
                ret.append(
                    ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                        health=providers_health.get(api, {}).get(
                            p.provider_id, HealthResponse(status=HealthStatus.NOT_IMPLEMENTED)
                        ),
                    )
                )

        return ListProvidersResponse(data=ret)

    async def list_routes(self) -> ListRoutesResponse:
        run_config = self.config.run_config

        ret = []
        all_endpoints = get_all_api_endpoints()
        for api, endpoints in all_endpoints.items():
            providers = run_config.providers.get(api.value, [])
            ret.extend(
                [
                    RouteInfo(
                        route=e.route,
                        method=e.method,
                        provider_types=[p.provider_type for p in providers],
                    )
                    for e in endpoints
                ]
            )

        return ListRoutesResponse(data=ret)

    async def get_providers_health(self) -> Dict[str, Dict[str, HealthResponse]]:
        providers_health = {}
        timeout = 3.0
        for impl in self.impls.values():
            # skip if no health method
            if not hasattr(impl, "health"):
                continue

            await impl.initialize()
            try:
                health = await asyncio.wait_for(impl.health(), timeout=timeout)
                providers_health[impl.__provider_spec__.api.name] = health
            except asyncio.TimeoutError:
                health = HealthResponse(
                    status=HealthStatus.ERROR, message=f"Health check timed out after {timeout} seconds"
                )

        return providers_health

    async def health(self) -> HealthInfo:
        return HealthInfo(status=HealthStatus.OK)

    async def version(self) -> VersionInfo:
        return VersionInfo(version=version("llama-stack"))

    async def shutdown(self) -> None:
        pass
