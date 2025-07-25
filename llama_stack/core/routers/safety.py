# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.inference import (
    Message,
)
from llama_stack.apis.safety import RunShieldResponse, Safety
from llama_stack.apis.shields import Shield
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

log = get_logger(name=__name__, category="core")


class SafetyRouter(Safety):
    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        log.debug("Initializing SafetyRouter")
        self.routing_table = routing_table

    async def initialize(self) -> None:
        log.debug("SafetyRouter.initialize")
        pass

    async def shutdown(self) -> None:
        log.debug("SafetyRouter.shutdown")
        pass

    async def register_shield(
        self,
        shield_id: str,
        provider_shield_id: str | None = None,
        provider_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Shield:
        log.debug(f"SafetyRouter.register_shield: {shield_id}")
        return await self.routing_table.register_shield(shield_id, provider_shield_id, provider_id, params)

    async def unregister_shield(self, identifier: str) -> None:
        logger.debug(f"SafetyRouter.unregister_shield: {identifier}")
        return await self.routing_table.unregister_shield(identifier)

    async def run_shield(
        self,
        shield_id: str,
        messages: list[Message],
        params: dict[str, Any] = None,
    ) -> RunShieldResponse:
        log.debug(f"SafetyRouter.run_shield: {shield_id}")
        provider = await self.routing_table.get_provider_impl(shield_id)
        return await provider.run_shield(
            shield_id=shield_id,
            messages=messages,
            params=params,
        )
