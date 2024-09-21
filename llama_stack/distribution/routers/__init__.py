# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Tuple

from llama_stack.distribution.datatypes import Api


async def get_router_impl(api: str, provider_routing_table: Dict[str, Any]):
    from .routers import InferenceRouter, MemoryRouter
    from .routing_table import RoutingTable

    api2routers = {
        "memory": MemoryRouter,
        "inference": InferenceRouter,
    }

    routing_table = RoutingTable(provider_routing_table)
    routing_table.print()

    impl = api2routers[api](routing_table)
    # impl = Router(api, provider_routing_table)
    await impl.initialize()
    return impl
