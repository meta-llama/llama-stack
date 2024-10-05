# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, List

from llama_stack.distribution.datatypes import *  # noqa: F403
from .routing_tables import (
    MemoryBanksRoutingTable,
    ModelsRoutingTable,
    RoutableObject,
    RoutedProtocol,
    ShieldsRoutingTable,
)


async def get_routing_table_impl(
    api: Api,
    registry: List[RoutableObject],
    impls_by_provider_id: Dict[str, RoutedProtocol],
    _deps,
) -> Any:
    api_to_tables = {
        "memory_banks": MemoryBanksRoutingTable,
        "models": ModelsRoutingTable,
        "shields": ShieldsRoutingTable,
    }
    if api.value not in api_to_tables:
        raise ValueError(f"API {api.value} not found in router map")

    impl = api_to_tables[api.value](registry, impls_by_provider_id)
    await impl.initialize()
    return impl


async def get_auto_router_impl(api: Api, routing_table: RoutingTable, _deps) -> Any:
    from .routers import InferenceRouter, MemoryRouter, SafetyRouter

    api_to_routers = {
        "memory": MemoryRouter,
        "inference": InferenceRouter,
        "safety": SafetyRouter,
    }
    if api.value not in api_to_routers:
        raise ValueError(f"API {api.value} not found in router map")

    impl = api_to_routers[api.value](routing_table)
    await impl.initialize()
    return impl
