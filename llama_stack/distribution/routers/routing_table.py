# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any, Dict, List

from llama_stack.distribution.datatypes import (
    Api,
    GenericProviderConfig,
    ProviderRoutingEntry,
)
from llama_stack.distribution.distribution import api_providers
from llama_stack.distribution.utils.dynamic import instantiate_provider
from termcolor import cprint


class RoutingTable:
    def __init__(self, provider_routing_table: Dict[str, List[ProviderRoutingEntry]]):
        self.provider_routing_table = provider_routing_table
        # map {api: {routing_key: impl}}, e.g. {'inference': {'8b': <MetaReferenceImpl>, '70b': <OllamaImpl>}}
        self.api2routes = {}

    async def initialize(self, api_str: str) -> None:
        """Initialize the routing table with concrete provider impls"""
        if api_str not in self.provider_routing_table:
            raise ValueError(f"API {api_str} not found in routing table")

        providers = api_providers()[Api(api_str)]
        routing_list = self.provider_routing_table[api_str]

        self.api2routes[api_str] = {}
        for rt_entry in routing_list:
            rt_key = rt_entry.routing_key
            provider_id = rt_entry.provider_id
            impl = await instantiate_provider(
                providers[provider_id],
                deps=[],
                provider_config=GenericProviderConfig(
                    provider_id=provider_id, config=rt_entry.config
                ),
            )
            cprint(f"impl = {impl}", "red")
            self.api2routes[api_str][rt_key] = impl

        cprint(f"> Initialized implementations for {api_str} in routing table", "blue")

    async def shutdown(self, api_str: str) -> None:
        """Shutdown the routing table"""
        if api_str not in self.api2routes:
            return

        for impl in self.api2routes[api_str].values():
            await impl.shutdown()

    def get_provider_impl(self, api: str, routing_key: str) -> Any:
        """Get the provider impl for a given api and routing key"""
        return self.api2routes[api][routing_key]
