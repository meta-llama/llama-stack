# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.distribution.datatypes import *  # noqa: F403

from llama_stack.distribution.store import DistributionRegistry

from .routing_tables import (
    DatasetsRoutingTable,
    MemoryBanksRoutingTable,
    ModelsRoutingTable,
    ScoringFunctionsRoutingTable,
    ShieldsRoutingTable,
)


async def get_routing_table_impl(
    api: Api,
    impls_by_provider_id: Dict[str, RoutedProtocol],
    _deps,
    dist_registry: DistributionRegistry,
) -> Any:
    api_to_tables = {
        "memory_banks": MemoryBanksRoutingTable,
        "models": ModelsRoutingTable,
        "shields": ShieldsRoutingTable,
        "datasets": DatasetsRoutingTable,
        "scoring_functions": ScoringFunctionsRoutingTable,
    }

    if api.value not in api_to_tables:
        raise ValueError(f"API {api.value} not found in router map")

    impl = api_to_tables[api.value](impls_by_provider_id, dist_registry)
    await impl.initialize()
    return impl


async def get_auto_router_impl(api: Api, routing_table: RoutingTable, _deps) -> Any:
    from .routers import (
        DatasetIORouter,
        InferenceRouter,
        MemoryRouter,
        SafetyRouter,
        ScoringRouter,
    )

    api_to_routers = {
        "memory": MemoryRouter,
        "inference": InferenceRouter,
        "safety": SafetyRouter,
        "datasetio": DatasetIORouter,
        "scoring": ScoringRouter,
    }
    if api.value not in api_to_routers:
        raise ValueError(f"API {api.value} not found in router map")

    impl = api_to_routers[api.value](routing_table)
    await impl.initialize()
    return impl
