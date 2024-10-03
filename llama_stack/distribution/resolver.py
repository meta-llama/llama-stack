# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import importlib

from typing import Any, Dict, List, Set

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.distribution import (
    builtin_automatically_routed_apis,
    get_provider_registry,
)
from llama_stack.distribution.inspect import DistributionInspectImpl
from llama_stack.distribution.utils.dynamic import instantiate_class_type


async def resolve_impls_with_routing(run_config: StackRunConfig) -> Dict[Api, Any]:
    """
    Does two things:
    - flatmaps, sorts and resolves the providers in dependency order
    - for each API, produces either a (local, passthrough or router) implementation
    """
    all_providers = get_provider_registry()
    specs = {}
    configs = {}

    for api_str, config in run_config.api_providers.items():
        api = Api(api_str)

        # TODO: check that these APIs are not in the routing table part of the config
        providers = all_providers[api]

        # skip checks for API whose provider config is specified in routing_table
        if isinstance(config, PlaceholderProviderConfig):
            continue

        if config.provider_type not in providers:
            raise ValueError(
                f"Provider `{config.provider_type}` is not available for API `{api}`"
            )
        specs[api] = providers[config.provider_type]
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

        if info.router_api.value not in run_config.routing_table:
            raise ValueError(f"Routing table for `{source_api.value}` is not provided?")

        routing_table = run_config.routing_table[info.router_api.value]

        providers = all_providers[info.router_api]

        inner_specs = []
        inner_deps = []
        for rt_entry in routing_table:
            if rt_entry.provider_type not in providers:
                raise ValueError(
                    f"Provider `{rt_entry.provider_type}` is not available for API `{api}`"
                )
            inner_specs.append(providers[rt_entry.provider_type])
            inner_deps.extend(providers[rt_entry.provider_type].api_dependencies)

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
        print(f"  {spec.api}: {spec.provider_type}")
    print("")
    impls = {}
    for spec in sorted_specs:
        api = spec.api
        deps = {api: impls[api] for api in spec.api_dependencies}
        impl = await instantiate_provider(spec, deps, configs[api])

        impls[api] = impl

    impls[Api.inspect] = DistributionInspectImpl()
    specs[Api.inspect] = InlineProviderSpec(
        api=Api.inspect,
        provider_type="__distribution_builtin__",
        config_class="",
        module="",
    )

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


# returns a class implementing the protocol corresponding to the Api
async def instantiate_provider(
    provider_spec: ProviderSpec,
    deps: Dict[str, Any],
    provider_config: Union[GenericProviderConfig, RoutingTable],
):
    module = importlib.import_module(provider_spec.module)

    args = []
    if isinstance(provider_spec, RemoteProviderSpec):
        if provider_spec.adapter:
            method = "get_adapter_impl"
        else:
            method = "get_client_impl"

        assert isinstance(provider_config, GenericProviderConfig)
        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider_config.config)
        args = [config, deps]
    elif isinstance(provider_spec, AutoRoutedProviderSpec):
        method = "get_auto_router_impl"

        config = None
        args = [provider_spec.api, deps[provider_spec.routing_table_api], deps]
    elif isinstance(provider_spec, RoutingTableProviderSpec):
        method = "get_routing_table_impl"

        assert isinstance(provider_config, List)
        routing_table = provider_config

        inner_specs = {x.provider_type: x for x in provider_spec.inner_specs}
        inner_impls = []
        for routing_entry in routing_table:
            impl = await instantiate_provider(
                inner_specs[routing_entry.provider_type],
                deps,
                routing_entry,
            )
            inner_impls.append((routing_entry.routing_key, impl))

        config = None
        args = [provider_spec.api, inner_impls, routing_table, deps]
    else:
        method = "get_provider_impl"

        assert isinstance(provider_config, GenericProviderConfig)
        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider_config.config)
        args = [config, deps]

    fn = getattr(module, method)
    impl = await fn(*args)
    impl.__provider_spec__ = provider_spec
    impl.__provider_config__ = config
    return impl
