# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
from typing import Any, Dict, List, Optional

import litellm
import requests

from llama_stack.apis.inference import Message
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
from llama_stack.providers.utils.inference.openai_compat import convert_message_to_openai_dict_new

from .config import SambaNovaSafetyConfig

logger = logging.getLogger(__name__)

CANNED_RESPONSE_TEXT = "I can't answer that. Can I help with something else?"


class SambaNovaSafetyAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: SambaNovaSafetyConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        list_models_url = self.config.url + "/models"
        try:
            response = requests.get(list_models_url)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request to {list_models_url} failed") from e
        available_models = [model.get("id") for model in response.json().get("data", {})]
        if (
            len(available_models) == 0
            or "guard" not in shield.provider_resource_id.lower()
            or shield.provider_resource_id.split("sambanova/")[-1] not in available_models
        ):
            raise ValueError(f"Shield {shield.provider_resource_id} not found in SambaNova")

    async def run_shield(
        self, shield_id: str, messages: List[Message], params: Optional[Dict[str, Any]] = None
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")

        shield_params = shield.params
        logger.debug(f"run_shield::{shield_params}::messages={messages}")
        content_messages = [await convert_message_to_openai_dict_new(m) for m in messages]
        logger.debug(f"run_shield::final:messages::{json.dumps(content_messages, indent=2)}:")

        response = litellm.completion(model=shield.provider_resource_id, messages=content_messages)
        shield_message = response.choices[0].message.content

        if "unsafe" in shield_message.lower():
            user_message = CANNED_RESPONSE_TEXT
            violation_type = shield_message.split("\n")[-1]
            metadata = {"violation_type": violation_type}

            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=user_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata=metadata,
                )
            )

        return RunShieldResponse()
