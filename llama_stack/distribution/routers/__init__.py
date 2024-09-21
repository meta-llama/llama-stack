# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Tuple

from llama_stack.distribution.datatypes import Api, ProviderRoutingEntry


async def get_router_impl(
    api: str, provider_routing_table: Dict[str, List[ProviderRoutingEntry]]
):
    from .routers import InferenceRouter, MemoryRouter
    from .routing_table import RoutingTable

    api2routers = {
        "memory": MemoryRouter,
        "inference": InferenceRouter,
    }

    # initialize routing table with concrete provider impls
    routing_table = RoutingTable(provider_routing_table)

    impl = api2routers[api](routing_table)
    await impl.initialize()
    return impl
