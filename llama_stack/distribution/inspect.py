# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from importlib.metadata import version
from typing import Dict, List

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

    async def list_providers(self) -> Dict[str, List[ProviderInfo]]:
        run_config = self.config.run_config

        ret = {}
        for api, providers in run_config.providers.items():
            ret[api] = [
                ProviderInfo(
                    provider_id=p.provider_id,
                    provider_type=p.provider_type,
                )
                for p in providers
            ]

        return ret

    async def list_routes(self) -> Dict[str, List[RouteInfo]]:
        run_config = self.config.run_config

        ret = {}
        all_endpoints = get_all_api_endpoints()
        for api, endpoints in all_endpoints.items():
            providers = run_config.providers.get(api.value, [])
            ret[api.value] = [
                RouteInfo(
                    route=e.route,
                    method=e.method,
                    provider_types=[p.provider_type for p in providers],
                )
                for e in endpoints
            ]
        return ret

    async def health(self) -> HealthInfo:
        return HealthInfo(status="OK")

    async def version(self) -> VersionInfo:
        return VersionInfo(version=version("llama-stack"))
