# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import importlib
import inspect
from typing import Any

from llama_stack.apis.agents import Agents
from llama_stack.apis.benchmarks import Benchmarks
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.eval import Eval
from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference, InferenceProvider
from llama_stack.apis.inspect import Inspect
from llama_stack.apis.models import Models
from llama_stack.apis.post_training import PostTraining
from llama_stack.apis.providers import Providers as ProvidersAPI
from llama_stack.apis.safety import Safety
from llama_stack.apis.scoring import Scoring
from llama_stack.apis.scoring_functions import ScoringFunctions
from llama_stack.apis.shields import Shields
from llama_stack.apis.telemetry import Telemetry
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_dbs import VectorDBs
from llama_stack.apis.vector_io import VectorIO
from llama_stack.distribution.client import get_client_impl
from llama_stack.distribution.datatypes import (
    AccessRule,
    AutoRoutedProviderSpec,
    Provider,
    RoutingTableProviderSpec,
    StackRunConfig,
)
from llama_stack.distribution.distribution import builtin_automatically_routed_apis, resolve_config
from llama_stack.distribution.store import DistributionRegistry
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import (
    Api,
    BenchmarksProtocolPrivate,
    DatasetsProtocolPrivate,
    ModelsProtocolPrivate,
    ProviderSpec,
    RemoteProviderConfig,
    RemoteProviderSpec,
    ScoringFunctionsProtocolPrivate,
    ShieldsProtocolPrivate,
    ToolGroupsProtocolPrivate,
    VectorDBsProtocolPrivate,
)

logger = get_logger(name=__name__, category="core")


class InvalidProviderError(Exception):
    pass


def api_protocol_map() -> dict[Api, Any]:
    return {
        Api.providers: ProvidersAPI,
        Api.agents: Agents,
        Api.inference: Inference,
        Api.inspect: Inspect,
        Api.vector_io: VectorIO,
        Api.vector_dbs: VectorDBs,
        Api.models: Models,
        Api.safety: Safety,
        Api.shields: Shields,
        Api.telemetry: Telemetry,
        Api.datasetio: DatasetIO,
        Api.datasets: Datasets,
        Api.scoring: Scoring,
        Api.scoring_functions: ScoringFunctions,
        Api.eval: Eval,
        Api.benchmarks: Benchmarks,
        Api.post_training: PostTraining,
        Api.tool_groups: ToolGroups,
        Api.tool_runtime: ToolRuntime,
        Api.files: Files,
    }


def api_protocol_map_for_compliance_check() -> dict[Api, Any]:
    return {
        **api_protocol_map(),
        Api.inference: InferenceProvider,
    }


def additional_protocols_map() -> dict[Api, Any]:
    return {
        Api.inference: (ModelsProtocolPrivate, Models, Api.models),
        Api.tool_groups: (ToolGroupsProtocolPrivate, ToolGroups, Api.tool_groups),
        Api.vector_io: (VectorDBsProtocolPrivate, VectorDBs, Api.vector_dbs),
        Api.safety: (ShieldsProtocolPrivate, Shields, Api.shields),
        Api.datasetio: (DatasetsProtocolPrivate, Datasets, Api.datasets),
        Api.scoring: (
            ScoringFunctionsProtocolPrivate,
            ScoringFunctions,
            Api.scoring_functions,
        ),
        Api.eval: (BenchmarksProtocolPrivate, Benchmarks, Api.benchmarks),
    }


# TODO: make all this naming far less atrocious. Provider. ProviderSpec. ProviderWithSpec. WTF!
class ProviderWithSpec(Provider):
    spec: ProviderSpec


ProviderRegistry = dict[Api, dict[str, ProviderSpec]]


async def resolve_impls(
    run_config: StackRunConfig,
    provider_registry: ProviderRegistry,
    dist_registry: DistributionRegistry,
    policy: list[AccessRule],
) -> dict[Api, Any]:
    """
    Resolves provider implementations by:
    1. Validating and organizing providers.
    2. Sorting them in dependency order.
    3. Instantiating them with required dependencies.
    """
    routing_table_apis = {x.routing_table_api for x in builtin_automatically_routed_apis()}
    router_apis = {x.router_api for x in builtin_automatically_routed_apis()}

    providers_with_specs = validate_and_prepare_providers(
        run_config, provider_registry, routing_table_apis, router_apis
    )

    apis_to_serve = run_config.apis or set(
        list(providers_with_specs.keys()) + [x.value for x in routing_table_apis] + [x.value for x in router_apis]
    )

    providers_with_specs.update(specs_for_autorouted_apis(apis_to_serve))

    sorted_providers = sort_providers_by_deps(providers_with_specs, run_config)

    return await instantiate_providers(sorted_providers, router_apis, dist_registry, run_config, policy)


def specs_for_autorouted_apis(apis_to_serve: list[str] | set[str]) -> dict[str, dict[str, ProviderWithSpec]]:
    """Generates specifications for automatically routed APIs."""
    specs = {}
    for info in builtin_automatically_routed_apis():
        if info.router_api.value not in apis_to_serve:
            continue

        specs[info.routing_table_api.value] = {
            "__builtin__": ProviderWithSpec(
                provider_id="__routing_table__",
                provider_type="__routing_table__",
                config=None,
                spec=RoutingTableProviderSpec(
                    api=info.routing_table_api,
                    router_api=info.router_api,
                    module="llama_stack.distribution.routers",
                    api_dependencies=[],
                    deps__=[f"inner-{info.router_api.value}"],
                ),
            )
        }

        specs[info.router_api.value] = {
            "__builtin__": ProviderWithSpec(
                provider_id="__autorouted__",
                provider_type="__autorouted__",
                config=None,
                spec=AutoRoutedProviderSpec(
                    api=info.router_api,
                    module="llama_stack.distribution.routers",
                    routing_table_api=info.routing_table_api,
                    api_dependencies=[info.routing_table_api],
                    # Add telemetry as an optional dependency to all auto-routed providers
                    optional_api_dependencies=[Api.telemetry],
                    deps__=([info.routing_table_api.value, Api.telemetry.value]),
                ),
            )
        }
    return specs


def validate_and_prepare_providers(
    run_config: StackRunConfig, provider_registry: ProviderRegistry, routing_table_apis: set[Api], router_apis: set[Api]
) -> dict[str, dict[str, ProviderWithSpec]]:
    """Validates providers, handles deprecations, and organizes them into a spec dictionary."""
    providers_with_specs: dict[str, dict[str, ProviderWithSpec]] = {}

    for api_str, providers in run_config.providers.items():
        api = Api(api_str)
        if api in routing_table_apis:
            raise ValueError(f"Provider for `{api_str}` is automatically provided and cannot be overridden")

        specs = {}
        for provider in providers:
            if not provider.provider_id or provider.provider_id == "__disabled__":
                logger.warning(f"Provider `{provider.provider_type}` for API `{api}` is disabled")
                continue

            validate_provider(provider, api, provider_registry)
            p = provider_registry[api][provider.provider_type]
            p.deps__ = [a.value for a in p.api_dependencies] + [a.value for a in p.optional_api_dependencies]
            spec = ProviderWithSpec(spec=p, **provider.model_dump())
            specs[provider.provider_id] = spec

        key = api_str if api not in router_apis else f"inner-{api_str}"
        providers_with_specs[key] = specs

    return providers_with_specs


def validate_provider(provider: Provider, api: Api, provider_registry: ProviderRegistry):
    """Validates if the provider is allowed and handles deprecations."""
    if provider.provider_type not in provider_registry[api]:
        raise ValueError(f"Provider `{provider.provider_type}` is not available for API `{api}`")

    p = provider_registry[api][provider.provider_type]
    if p.deprecation_error:
        logger.error(p.deprecation_error)
        raise InvalidProviderError(p.deprecation_error)
    elif p.deprecation_warning:
        logger.warning(
            f"Provider `{provider.provider_type}` for API `{api}` is deprecated and will be removed in a future release: {p.deprecation_warning}",
        )


def sort_providers_by_deps(
    providers_with_specs: dict[str, dict[str, ProviderWithSpec]], run_config: StackRunConfig
) -> list[tuple[str, ProviderWithSpec]]:
    """Sorts providers based on their dependencies."""
    sorted_providers: list[tuple[str, ProviderWithSpec]] = topological_sort(
        {k: list(v.values()) for k, v in providers_with_specs.items()}
    )

    logger.debug(f"Resolved {len(sorted_providers)} providers")
    for api_str, provider in sorted_providers:
        logger.debug(f" {api_str} => {provider.provider_id}")
    return sorted_providers


async def instantiate_providers(
    sorted_providers: list[tuple[str, ProviderWithSpec]],
    router_apis: set[Api],
    dist_registry: DistributionRegistry,
    run_config: StackRunConfig,
    policy: list[AccessRule],
) -> dict:
    """Instantiates providers asynchronously while managing dependencies."""
    impls: dict[Api, Any] = {}
    inner_impls_by_provider_id: dict[str, dict[str, Any]] = {f"inner-{x.value}": {} for x in router_apis}
    for api_str, provider in sorted_providers:
        # Skip providers that are not enabled
        if provider.provider_id is None:
            continue

        deps = {a: impls[a] for a in provider.spec.api_dependencies}
        for a in provider.spec.optional_api_dependencies:
            if a in impls:
                deps[a] = impls[a]

        inner_impls = {}
        if isinstance(provider.spec, RoutingTableProviderSpec):
            inner_impls = inner_impls_by_provider_id[f"inner-{provider.spec.router_api.value}"]

        impl = await instantiate_provider(provider, deps, inner_impls, dist_registry, run_config, policy)

        if api_str.startswith("inner-"):
            inner_impls_by_provider_id[api_str][provider.provider_id] = impl
        else:
            api = Api(api_str)
            impls[api] = impl

    return impls


def topological_sort(
    providers_with_specs: dict[str, list[ProviderWithSpec]],
) -> list[tuple[str, ProviderWithSpec]]:
    def dfs(kv, visited: set[str], stack: list[str]):
        api_str, providers = kv
        visited.add(api_str)

        deps = []
        for provider in providers:
            for dep in provider.spec.deps__:
                deps.append(dep)

        for dep in deps:
            if dep not in visited and dep in providers_with_specs:
                dfs((dep, providers_with_specs[dep]), visited, stack)

        stack.append(api_str)

    visited: set[str] = set()
    stack: list[str] = []

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
    deps: dict[Api, Any],
    inner_impls: dict[str, Any],
    dist_registry: DistributionRegistry,
    run_config: StackRunConfig,
    policy: list[AccessRule],
):
    provider_spec = provider.spec
    if not hasattr(provider_spec, "module"):
        raise AttributeError(f"ProviderSpec of type {type(provider_spec)} does not have a 'module' attribute")

    logger.debug(f"Instantiating provider {provider.provider_id} from {provider_spec.module}")
    module = importlib.import_module(provider_spec.module)
    args = []
    if isinstance(provider_spec, RemoteProviderSpec):
        config = resolve_config(provider=provider, provider_spec=provider_spec)

        method = "get_adapter_impl"
        args = [config, deps]

    elif isinstance(provider_spec, AutoRoutedProviderSpec):
        method = "get_auto_router_impl"

        config = None
        args = [provider_spec.api, deps[provider_spec.routing_table_api], deps, run_config, policy]
    elif isinstance(provider_spec, RoutingTableProviderSpec):
        method = "get_routing_table_impl"

        config = None
        args = [provider_spec.api, inner_impls, deps, dist_registry, policy]
    else:
        method = "get_provider_impl"

        config = resolve_config(provider=provider, provider_spec=provider_spec)
        args = [config, deps]
        if "policy" in inspect.signature(getattr(module, method)).parameters:
            args.append(policy)

    fn = getattr(module, method)
    impl = await fn(*args)
    impl.__provider_id__ = provider.provider_id
    impl.__provider_spec__ = provider_spec
    impl.__provider_config__ = config

    protocols = api_protocol_map_for_compliance_check()
    additional_protocols = additional_protocols_map()
    # TODO: check compliance for special tool groups
    # the impl should be for Api.tool_runtime, the name should be the special tool group, the protocol should be the special tool group protocol
    check_protocol_compliance(impl, protocols[provider_spec.api])
    if not isinstance(provider_spec, AutoRoutedProviderSpec) and provider_spec.api in additional_protocols:
        additional_api, _, _ = additional_protocols[provider_spec.api]
        check_protocol_compliance(impl, additional_api)

    return impl


def check_protocol_compliance(obj: Any, protocol: Any) -> None:
    missing_methods = []

    mro = type(obj).__mro__
    for name, value in inspect.getmembers(protocol):
        if inspect.isfunction(value) and hasattr(value, "__webmethod__"):
            if value.__webmethod__.experimental:
                continue
            if not hasattr(obj, name):
                missing_methods.append((name, "missing"))
            elif not callable(getattr(obj, name)):
                missing_methods.append((name, "not_callable"))
            else:
                # Check if the method signatures are compatible
                obj_method = getattr(obj, name)
                proto_sig = inspect.signature(value)
                obj_sig = inspect.signature(obj_method)

                proto_params = set(proto_sig.parameters)
                proto_params.discard("self")
                obj_params = set(obj_sig.parameters)
                obj_params.discard("self")
                if not (proto_params <= obj_params):
                    logger.error(f"Method {name} incompatible proto: {proto_params} vs. obj: {obj_params}")
                    missing_methods.append((name, "signature_mismatch"))
                else:
                    # Check if the method has a concrete implementation (not just a protocol stub)
                    # Find all classes in MRO that define this method
                    method_owners = [cls for cls in mro if name in cls.__dict__]

                    # Allow methods from mixins/parents, only reject if ONLY the protocol defines it
                    if len(method_owners) == 1 and method_owners[0].__name__ == protocol.__name__:
                        # Only reject if the method is ONLY defined in the protocol itself (abstract stub)
                        missing_methods.append((name, "not_actually_implemented"))

    if missing_methods:
        raise ValueError(
            f"Provider `{obj.__provider_id__} ({obj.__provider_spec__.api})` does not implement the following methods:\n{missing_methods}"
        )


async def resolve_remote_stack_impls(
    config: RemoteProviderConfig,
    apis: list[str],
) -> dict[Api, Any]:
    protocols = api_protocol_map()
    additional_protocols = additional_protocols_map()

    impls = {}
    for api_str in apis:
        api = Api(api_str)
        impls[api] = await get_client_impl(
            protocols[api],
            config,
            {},
        )
        if api in additional_protocols:
            _, additional_protocol, additional_api = additional_protocols[api]
            impls[additional_api] = await get_client_impl(
                additional_protocol,
                config,
                {},
            )

    return impls
