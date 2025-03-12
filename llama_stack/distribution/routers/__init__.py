# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.distribution.datatypes import RoutedProtocol
from llama_stack.distribution.store import DistributionRegistry
from llama_stack.providers.datatypes import Api, RoutingTable

from .routing_tables import (
    BenchmarksRoutingTable,
    DatasetsRoutingTable,
    ModelsRoutingTable,
    ScoringFunctionsRoutingTable,
    ShieldsRoutingTable,
    ToolGroupsRoutingTable,
    VectorDBsRoutingTable,
)


async def get_routing_table_impl(
    api: Api,
    impls_by_provider_id: Dict[str, RoutedProtocol],
    _deps,
    dist_registry: DistributionRegistry,
) -> Any:
    api_to_tables = {
        "vector_dbs": VectorDBsRoutingTable,
        "models": ModelsRoutingTable,
        "shields": ShieldsRoutingTable,
        "datasets": DatasetsRoutingTable,
        "scoring_functions": ScoringFunctionsRoutingTable,
        "benchmarks": BenchmarksRoutingTable,
        "tool_groups": ToolGroupsRoutingTable,
    }

    if api.value not in api_to_tables:
        raise ValueError(f"API {api.value} not found in router map")

    impl = api_to_tables[api.value](impls_by_provider_id, dist_registry)
    await impl.initialize()
    return impl


async def get_auto_router_impl(api: Api, routing_table: RoutingTable, deps: Dict[str, Any]) -> Any:
    from .routers import (
        DatasetIORouter,
        EvalRouter,
        InferenceRouter,
        SafetyRouter,
        ScoringRouter,
        ToolRuntimeRouter,
        VectorIORouter,
    )

    api_to_routers = {
        "vector_io": VectorIORouter,
        "inference": InferenceRouter,
        "safety": SafetyRouter,
        "datasetio": DatasetIORouter,
        "scoring": ScoringRouter,
        "eval": EvalRouter,
        "tool_runtime": ToolRuntimeRouter,
    }
    api_to_deps = {
        "inference": {"telemetry": Api.telemetry},
    }
    if api.value not in api_to_routers:
        raise ValueError(f"API {api.value} not found in router map")

    api_to_dep_impl = {}
    for dep_name, dep_api in api_to_deps.get(api.value, {}).items():
        if dep_api in deps:
            api_to_dep_impl[dep_name] = deps[dep_api]

    impl = api_to_routers[api.value](routing_table, **api_to_dep_impl)
    await impl.initialize()
    return impl
