# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import Any, Dict, List

import httpx

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import RunShieldResponse, Safety
from llama_stack.apis.safety.safety import SafetyViolation, ViolationLevel
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate

from .config import FiddlecubeSafetyConfig

logger = logging.getLogger(__name__)


class FiddlecubeSafetyAdapter(Safety, ShieldsProtocolPrivate):
    """
    Implementation of the Fiddlecube Safety API.

    Accepts a list of messages for content moderation.
    Optionally, a list of `excluded_categories` can be provided to exclude certain categories from moderation.
    Category name strings map to the llama-guard risk taxonomy.
    """

    def __init__(self, config: FiddlecubeSafetyConfig) -> None:
        self.config = config
        self.registered_shields = []

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        pass

    async def run_shield(
        self, shield_id: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        async with httpx.AsyncClient(timeout=30.0) as client:
            request_body = {
                "messages": [message.model_dump(mode="json") for message in messages],
            }
            if params and params.get("excluded_categories"):
                request_body["excluded_categories"] = params.get("excluded_categories")
            headers = {"Content-Type": "application/json"}
            response = await client.post(
                f"{self.config.api_url}/safety/guard/check",
                json=request_body,
                headers=headers,
            )

        if response.status_code != 200:
            logger.error(f"FiddleCube API error: {response.status_code} - {response.text}")
            raise RuntimeError("Failed to run shield with FiddleCube API")

        response_data = response.json()

        if response_data.get("violation"):
            violation = response_data.get("violation")
            user_message = violation.get("user_message")
            metadata = violation.get("metadata")
            violation_level = ViolationLevel(violation.get("violation_level"))
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=user_message,
                    violation_level=violation_level,
                    metadata=metadata,
                )
            )

        return RunShieldResponse()
