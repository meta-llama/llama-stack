# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Set

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.distribution import (
    api_providers,
    builtin_automatically_routed_apis,
)
from llama_stack.distribution.utils.dynamic import instantiate_provider


async def resolve_impls_with_routing(run_config: StackRunConfig) -> Dict[Api, Any]:
    """
    Does two things:
    - flatmaps, sorts and resolves the providers in dependency order
    - for each API, produces either a (local, passthrough or router) implementation
    """
    all_providers = api_providers()
    specs = {}
    configs = {}

    for api_str, config in run_config.api_providers.items():
        api = Api(api_str)

        # TODO: check that these APIs are not in the routing table part of the config
        providers = all_providers[api]

        # skip checks for API whose provider config is specified in routing_table
        if isinstance(config, PlaceholderProviderConfig):
            continue

        if config.provider_id not in providers:
            raise ValueError(
                f"Unknown provider `{config.provider_id}` is not available for API `{api}`"
            )
        specs[api] = providers[config.provider_id]
        configs[api] = config

    apis_to_serve = run_config.apis_to_serve or set(
        list(specs.keys()) + list(run_config.routing_table.keys())
    )
    for info in builtin_automatically_routed_apis():
        source_api = info.routing_table_api

        assert (
            source_api not in specs
        ), f"Routing table API {source_api} specified in wrong place?"
        assert (
            info.router_api not in specs
        ), f"Auto-routed API {info.router_api} specified in wrong place?"

        if info.router_api.value not in apis_to_serve:
            continue

        print("router_api", info.router_api)
        if info.router_api.value not in run_config.routing_table:
            raise ValueError(f"Routing table for `{source_api.value}` is not provided?")

        routing_table = run_config.routing_table[info.router_api.value]

        providers = all_providers[info.router_api]

        inner_specs = []
        inner_deps = []
        for rt_entry in routing_table:
            if rt_entry.provider_id not in providers:
                raise ValueError(
                    f"Unknown provider `{rt_entry.provider_id}` is not available for API `{api}`"
                )
            inner_specs.append(providers[rt_entry.provider_id])
            inner_deps.extend(providers[rt_entry.provider_id].api_dependencies)

        specs[source_api] = RoutingTableProviderSpec(
            api=source_api,
            module="llama_stack.distribution.routers",
            api_dependencies=inner_deps,
            inner_specs=inner_specs,
        )
        configs[source_api] = routing_table

        specs[info.router_api] = AutoRoutedProviderSpec(
            api=info.router_api,
            module="llama_stack.distribution.routers",
            routing_table_api=source_api,
            api_dependencies=[source_api],
        )
        configs[info.router_api] = {}

    sorted_specs = topological_sort(specs.values())
    print(f"Resolved {len(sorted_specs)} providers in topological order")
    for spec in sorted_specs:
        print(f"  {spec.api}: {spec.provider_id}")
    print("")
    impls = {}
    for spec in sorted_specs:
        api = spec.api
        deps = {api: impls[api] for api in spec.api_dependencies}
        impl = await instantiate_provider(spec, deps, configs[api])

        impls[api] = impl

    return impls, specs


def topological_sort(providers: List[ProviderSpec]) -> List[ProviderSpec]:
    by_id = {x.api: x for x in providers}

    def dfs(a: ProviderSpec, visited: Set[Api], stack: List[Api]):
        visited.add(a.api)

        for api in a.api_dependencies:
            if api not in visited:
                dfs(by_id[api], visited, stack)

        stack.append(a.api)

    visited = set()
    stack = []

    for a in providers:
        if a.api not in visited:
            dfs(a, visited, stack)

    return [by_id[x] for x in stack]
