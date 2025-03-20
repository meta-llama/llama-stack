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
    ListRoutesResponse,
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
        return HealthInfo(status="OK")

    async def version(self) -> VersionInfo:
        return VersionInfo(version=version("llama-stack"))

    async def shutdown(self) -> None:
        pass
