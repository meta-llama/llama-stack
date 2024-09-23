# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, List, Tuple

from llama_stack.distribution.datatypes import *  # noqa: F403


async def get_routing_table_impl(
    api: Api,
    inner_impls: List[Tuple[str, Any]],
    routing_table_config: Dict[str, List[RoutableProviderConfig]],
    _deps,
) -> Any:
    from .routing_tables import (
        MemoryBanksRoutingTable,
        ModelsRoutingTable,
        ShieldsRoutingTable,
    )

    api_to_tables = {
        "memory_banks": MemoryBanksRoutingTable,
        "models": ModelsRoutingTable,
        "shields": ShieldsRoutingTable,
    }
    if api.value not in api_to_tables:
        raise ValueError(f"API {api.value} not found in router map")

    impl = api_to_tables[api.value](inner_impls, routing_table_config)
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
