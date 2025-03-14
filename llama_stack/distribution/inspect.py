# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from importlib.metadata import version

from pydantic import BaseModel

from llama_stack.apis.inspect import (
    HealthInfo,
    Inspect,
    ProviderInfo,
    RouteInfo,
    VersionInfo,
)
from llama_stack.distribution.datatypes import StackRunConfig
from llama_stack.distribution.server.endpoints import get_all_api_endpoints


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

    async def initialize(self) -> None:
        pass

    async def list_providers(self) -> list[ProviderInfo]:
        return [
            ProviderInfo(
                api=api,
                provider_id=p.provider_id,
                provider_type=p.provider_type,
            )
            for api, providers in self.config.run_config.providers.items()
            for p in providers
        ]

    async def list_routes(self) -> list[RouteInfo]:
        return [
            RouteInfo(
                route=e.route,
                method=e.method,
                provider_types=[p.provider_type for p in self.config.run_config.providers.get(api.value, [])],
            )
            for api, endpoints in get_all_api_endpoints().items()
            for e in endpoints
        ]

    async def health(self) -> HealthInfo:
        return HealthInfo(status="OK")

    async def version(self) -> VersionInfo:
        return VersionInfo(version=version("llama-stack"))

    async def shutdown(self) -> None:
        pass
