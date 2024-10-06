# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict, List
from llama_stack.apis.inspect import *  # noqa: F403


from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.server.endpoints import get_all_api_endpoints
from llama_stack.providers.datatypes import *  # noqa: F403


def is_passthrough(spec: ProviderSpec) -> bool:
    return isinstance(spec, RemoteProviderSpec) and spec.adapter is None


class DistributionInspectImpl(Inspect):
    def __init__(self):
        pass

    async def list_providers(self) -> Dict[str, List[ProviderInfo]]:
        ret = {}
        all_providers = get_provider_registry()
        for api, providers in all_providers.items():
            ret[api.value] = [
                ProviderInfo(
                    provider_type=p.provider_type,
                    description="Passthrough" if is_passthrough(p) else "",
                )
                for p in providers.values()
            ]

        return ret

    async def list_routes(self) -> Dict[str, List[RouteInfo]]:
        ret = {}
        all_endpoints = get_all_api_endpoints()

        for api, endpoints in all_endpoints.items():
            ret[api.value] = [
                RouteInfo(
                    route=e.route,
                    method=e.method,
                    providers=[],
                )
                for e in endpoints
            ]
        return ret

    async def health(self) -> HealthInfo:
        return HealthInfo(status="OK")
