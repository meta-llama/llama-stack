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
from llama_stack.distribution.utils.dynamic import instantiate_class_type


# TODO: make all this naming far less atrocious. Provider. ProviderSpec. ProviderWithSpec. WTF!
class ProviderWithSpec(Provider):
    spec: ProviderSpec


# TODO: this code is not very straightforward to follow and needs one more round of refactoring
async def resolve_impls_with_routing(run_config: StackRunConfig) -> Dict[Api, Any]:
    """
    Does two things:
    - flatmaps, sorts and resolves the providers in dependency order
    - for each API, produces either a (local, passthrough or router) implementation
    """
    all_api_providers = get_provider_registry()

    routing_table_apis = set(
        x.routing_table_api for x in builtin_automatically_routed_apis()
    )
    router_apis = set(x.router_api for x in builtin_automatically_routed_apis())

    providers_with_specs = {}

    for api_str, providers in run_config.providers.items():
        api = Api(api_str)
        if api in routing_table_apis:
            raise ValueError(
                f"Provider for `{api_str}` is automatically provided and cannot be overridden"
            )

        specs = {}
        for provider in providers:
            if provider.provider_type not in all_api_providers[api]:
                raise ValueError(
                    f"Provider `{provider.provider_type}` is not available for API `{api}`"
                )

            spec = ProviderWithSpec(
                spec=all_api_providers[api][provider.provider_type],
                **(provider.dict()),
            )
            specs[provider.provider_id] = spec

        key = api_str if api not in router_apis else f"inner-{api_str}"
        providers_with_specs[key] = specs

    apis_to_serve = run_config.apis or set(
        list(providers_with_specs.keys()) + list(routing_table_apis)
    )

    for info in builtin_automatically_routed_apis():
        if info.router_api.value not in apis_to_serve:
            continue

        available_providers = providers_with_specs[f"inner-{info.router_api.value}"]

        inner_deps = []
        registry = getattr(run_config, info.routing_table_api.value)
        for entry in registry:
            if entry.provider_id not in available_providers:
                raise ValueError(
                    f"Provider `{entry.provider_id}` not found. Available providers: {list(available_providers.keys())}"
                )

            provider = available_providers[entry.provider_id]
            inner_deps.extend(provider.spec.api_dependencies)

        providers_with_specs[info.routing_table_api.value] = {
            "__builtin__": ProviderWithSpec(
                provider_id="__builtin__",
                provider_type="__routing_table__",
                config={},
                spec=RoutingTableProviderSpec(
                    api=info.routing_table_api,
                    router_api=info.router_api,
                    registry=registry,
                    module="llama_stack.distribution.routers",
                    api_dependencies=inner_deps,
                ),
            )
        }

        providers_with_specs[info.router_api.value] = {
            "__builtin__": ProviderWithSpec(
                provider_id="__builtin__",
                provider_type="__autorouted__",
                config={},
                spec=AutoRoutedProviderSpec(
                    api=info.router_api,
                    module="llama_stack.distribution.routers",
                    routing_table_api=info.routing_table_api,
                    api_dependencies=[info.routing_table_api],
                ),
            )
        }

    sorted_providers = topological_sort(
        {k: v.values() for k, v in providers_with_specs.items()}
    )
    sorted_providers.append(
        (
            "inspect",
            ProviderWithSpec(
                provider_id="__builtin__",
                provider_type="__builtin__",
                config={},
                spec=InlineProviderSpec(
                    api=Api.inspect,
                    provider_type="__builtin__",
                    config_class="llama_stack.distribution.inspect.DistributionInspectConfig",
                    module="llama_stack.distribution.inspect",
                ),
            ),
        )
    )

    print(f"Resolved {len(sorted_providers)} providers in topological order")
    for api_str, provider in sorted_providers:
        print(f"  {api_str}: ({provider.provider_id}) {provider.spec.provider_type}")
    print("")

    impls = {}
    inner_impls_by_provider_id = {f"inner-{x.value}": {} for x in router_apis}
    for api_str, provider in sorted_providers:
        deps = {a: impls[a] for a in provider.spec.api_dependencies}

        inner_impls = {}
        if isinstance(provider.spec, RoutingTableProviderSpec):
            for entry in provider.spec.registry:
                inner_impls[entry.provider_id] = inner_impls_by_provider_id[
                    f"inner-{provider.spec.router_api.value}"
                ][entry.provider_id]

        impl = await instantiate_provider(
            provider,
            deps,
            inner_impls,
        )
        if "inner-" in api_str:
            inner_impls_by_provider_id[api_str][provider.provider_id] = impl
        else:
            api = Api(api_str)
            impls[api] = impl

    return impls


def topological_sort(
    providers_with_specs: Dict[str, List[ProviderWithSpec]],
) -> List[ProviderWithSpec]:
    def dfs(kv, visited: Set[str], stack: List[str]):
        api_str, providers = kv
        visited.add(api_str)

        deps = []
        for provider in providers:
            for dep in provider.spec.api_dependencies:
                deps.append(dep.value)
            if isinstance(provider, AutoRoutedProviderSpec):
                deps.append(f"inner-{provider.api}")

        for dep in deps:
            if dep not in visited:
                dfs((dep, providers_with_specs[dep]), visited, stack)

        stack.append(api_str)

    visited = set()
    stack = []

    for api_str, providers in providers_with_specs.items():
        if api_str not in visited:
            dfs((api_str, providers), visited, stack)

    flattened = []
    for api_str in stack:
        for provider in providers_with_specs[api_str]:
            flattened.append((api_str, provider))
    return flattened


# returns a class implementing the protocol corresponding to the Api
async def instantiate_provider(
    provider: ProviderWithSpec,
    deps: Dict[str, Any],
    inner_impls: Dict[str, Any],
):
    provider_spec = provider.spec
    module = importlib.import_module(provider_spec.module)

    args = []
    if isinstance(provider_spec, RemoteProviderSpec):
        if provider_spec.adapter:
            method = "get_adapter_impl"
        else:
            method = "get_client_impl"

        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider.config)
        args = [config, deps]
    elif isinstance(provider_spec, AutoRoutedProviderSpec):
        method = "get_auto_router_impl"

        config = None
        args = [provider_spec.api, deps[provider_spec.routing_table_api], deps]
    elif isinstance(provider_spec, RoutingTableProviderSpec):
        method = "get_routing_table_impl"

        config = None
        args = [provider_spec.api, provider_spec.registry, inner_impls, deps]
    else:
        method = "get_provider_impl"

        config_type = instantiate_class_type(provider_spec.config_class)
        config = config_type(**provider.config)
        args = [config, deps]

    fn = getattr(module, method)
    impl = await fn(*args)
    impl.__provider_spec__ = provider_spec
    impl.__provider_config__ = config
    return impl
