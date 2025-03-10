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
    ListProvidersResponse,
    ListRoutesResponse,
    ProviderInfo,
    RouteInfo,
    VersionInfo,
)
from llama_stack.distribution.datatypes import StackRunConfig
from llama_stack.distribution.server.endpoints import get_all_api_endpoints
from llama_stack.providers.datatypes import HealthResponse, HealthStatus


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

    async def list_providers(self) -> ListProvidersResponse:
        run_config: StackRunConfig = self.config.run_config
        ret = []
        for api, providers in run_config.providers.items():
            for p in providers:
                ret.append(
                    ProviderInfo(
                        api=api,
                        provider_id=p.provider_id,
                        provider_type=p.provider_type,
                        # This endpoint will soon be deprecated so we don't need to health here
                        # See: https://github.com/meta-llama/llama-stack/issues/1623
                        health=HealthResponse(
                            status=HealthStatus.NOT_IMPLEMENTED,
                            message="See the '/providers' endpoint for health status",
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

    async def health(self) -> HealthInfo:
        return HealthInfo(status=HealthStatus.OK)

    async def version(self) -> VersionInfo:
        return VersionInfo(version=version("llama-stack"))

    async def shutdown(self) -> None:
        pass
