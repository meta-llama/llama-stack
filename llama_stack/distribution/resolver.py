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


async def resolve_impls_with_routing(run_config: StackRunConfig) -> Dict[Api, Any]:
    """
    Does two things:
    - flatmaps, sorts and resolves the providers in dependency order
    - for each API, produces either a (local, passthrough or router) implementation
    """
    all_api_providers = get_provider_registry()

    auto_routed_apis = builtin_automatically_routed_apis()
    providers_with_specs = {}

    for api_str, instances in run_config.providers.items():
        api = Api(api_str)
        if api in [a.routing_table_api for a in auto_routed_apis]:
            raise ValueError(
                f"Provider for `{api_str}` is automatically provided and cannot be overridden"
            )

        providers_with_specs[api] = {}
        for config in instances:
            if config.provider_type not in all_api_providers[api]:
                raise ValueError(
                    f"Provider `{config.provider_type}` is not available for API `{api}`"
                )

            spec = ProviderWithSpec(
                spec=all_api_providers[api][config.provider_type],
                **config,
            )
            providers_with_specs[api][spec.provider_id] = spec

    apis_to_serve = run_config.apis_to_serve or set(
        list(providers_with_specs.keys())
        + [a.routing_table_api.value for a in auto_routed_apis]
    )
    for info in builtin_automatically_routed_apis():
        if info.router_api.value not in apis_to_serve:
            continue

        if info.routing_table_api.value not in run_config:
            raise ValueError(
                f"Registry for `{info.routing_table_api.value}` is not provided?"
            )

        available_providers = providers_with_specs[info.router_api]

        inner_deps = []
        registry = run_config[info.routing_table_api.value]
        for entry in registry:
            if entry.provider_id not in available_providers:
                raise ValueError(
                    f"Provider `{entry.provider_id}` not found. Available providers: {list(available_providers.keys())}"
                )

            provider = available_providers[entry.provider_id]
            inner_deps.extend(provider.spec.api_dependencies)

        providers_with_specs[info.routing_table_api] = {
            "__builtin__": [
                ProviderWithSpec(
                    provider_id="__builtin__",
                    provider_type="__builtin__",
                    config=registry,
                    spec=RoutingTableProviderSpec(
                        api=info.routing_table_api,
                        router_api=info.router_api,
                        module="llama_stack.distribution.routers",
                        api_dependencies=inner_deps,
                    ),
                )
            ]
        }

        providers_with_specs[info.router_api] = {
            "__builtin__": [
                ProviderWithSpec(
                    provider_id="__builtin__",
                    provider_type="__builtin__",
                    config={},
                    spec=AutoRoutedProviderSpec(
                        api=info.router_api,
                        module="llama_stack.distribution.routers",
                        routing_table_api=source_api,
                        api_dependencies=[source_api],
                    ),
                )
            ]
        }

    sorted_providers = topological_sort(providers_with_specs)
    sorted_providers.append(
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
        )
    )

    print(f"Resolved {len(sorted_providers)} providers in topological order")
    for provider in sorted_providers:
        print(
            f"  {provider.spec.api}: ({provider.provider_id}) {provider.spec.provider_type}"
        )
    print("")
    impls = {}

    impls_by_provider_id = {}
    for provider in sorted_providers:
        api = provider.spec.api
        if api not in impls_by_provider_id:
            impls_by_provider_id[api] = {}

        deps = {api: impls[api] for api in provider.spec.api_dependencies}

        inner_impls = {}
        if isinstance(provider.spec, RoutingTableProviderSpec):
            for entry in provider.config:
                inner_impls[entry.provider_id] = impls_by_provider_id[
                    provider.spec.router_api
                ][entry.provider_id]

        impl = await instantiate_provider(
            provider,
            deps,
            inner_impls,
        )

        impls[api] = impl
        impls_by_provider_id[api][provider.provider_id] = impl

    return impls


def topological_sort(
    providers_with_specs: Dict[Api, List[ProviderWithSpec]],
) -> List[ProviderWithSpec]:
    def dfs(kv, visited: Set[Api], stack: List[Api]):
        api, providers = kv
        visited.add(api)

        deps = [dep for x in providers for dep in x.api_dependencies]
        for api in deps:
            if api not in visited:
                dfs((api, providers_with_specs[api]), visited, stack)

        stack.append(api)

    visited = set()
    stack = []

    for api, providers in providers_with_specs.items():
        if api not in visited:
            dfs((api, providers), visited, stack)

    flattened = []
    for api in stack:
        flattened.extend(providers_with_specs[api])
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

        assert isinstance(provider_config, list)
        registry = provider_config

        config = None
        args = [provider_spec.api, registry, inner_impls, deps]
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
