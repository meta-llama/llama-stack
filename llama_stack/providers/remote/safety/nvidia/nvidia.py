# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
from typing import Any, Dict, List

from llama_stack.apis.inference import Message, UserMessage
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.apis.shields import Shield
from llama_stack.distribution.library_client import convert_pydantic_to_json_value
from llama_stack.models.llama.datatypes import CoreModelId
from llama_stack.providers.datatypes import ShieldsProtocolPrivate
import requests

from .config import NVIDIASafetyConfig

logger = logging.getLogger(__name__)

SHIELD_IDS_TO_MODEL_MAPPING = {
    CoreModelId.llama3_8b_instruct.value: "meta/llama3-8b-instruct",
    CoreModelId.llama3_70b_instruct.value:  "meta/llama3-70b-instruct",
    CoreModelId.llama3_1_8b_instruct.value: "meta/llama-3.1-8b-instruct",
    CoreModelId.llama3_1_70b_instruct.value: "meta/llama-3.1-70b-instruct",
    CoreModelId.llama3_1_405b_instruct.value: "meta/llama-3.1-405b-instruct",
    CoreModelId.llama3_2_1b_instruct.value: "meta/llama-3.2-1b-instruct",
    CoreModelId.llama3_2_3b_instruct.value: "meta/llama-3.2-3b-instruct",
    CoreModelId.llama3_2_11b_vision_instruct.value: "meta/llama-3.2-11b-vision-instruct",
    CoreModelId.llama3_2_90b_vision_instruct.value: "meta/llama-3.2-90b-vision-instruct"
}

class NVIDIASafetyAdapter(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: NVIDIASafetyConfig) -> None:
        print(f"Initializing NVIDIASafetyAdapter({config.guardrails_service_url})...")
        self.config = config
        self.registered_shields = []

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        if not shield.provider_resource_id:
            raise ValueError(f"Shield model not provided. ")

    async def run_shield(
        self, shield_id: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")
        self.shield = NeMoGuardrails(self.config, SHIELD_IDS_TO_MODEL_MAPPING[shield.shield_id])
        return await self.shield.run(messages)
        
    
class NeMoGuardrails:
    def __init__(
        self,
        config: NVIDIASafetyConfig,
        model: str,
        threshold: float = 0.9,
        temperature: float = 1.0,
    ):
        self.config_id = config.config_id
        self.config_store_path = config.config_store_path
        self.model = model
        assert  self.config_id is not None or self.config_store_path is not None, "Must provide one of config id or config store path"
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0")

        self.config = config
        self.temperature = temperature
        self.threshold = threshold
        self.guardrails_service_url = config.guardrails_service_url

    async def run(self, messages: List[Message]) -> RunShieldResponse:
        headers = {
            "Accept": "application/json",
        }
        request_data = {
            "model": self.model,
            "messages": convert_pydantic_to_json_value(messages),
            "temperature": self.temperature,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "max_tokens": 160,
            "stream": False,
            "guardrails": {
                "config_id": self.config_id,
            }
        }
        response = requests.post(
            url=f"{self.guardrails_service_url}/v1/guardrail/checks",
            headers=headers,
            json=request_data
        )
        print(response)
        response.raise_for_status()
        if 'Content-Type' in response.headers and response.headers['Content-Type'].startswith('application/json'):
            response_json = response.json()
        if response_json["status"] == "blocked":
            user_message = "Sorry I cannot do this."
            metadata = response_json["rails_status"]
            
            return RunShieldResponse(
                violation=SafetyViolation(
                    user_message=user_message,
                    violation_level=ViolationLevel.ERROR,
                    metadata=metadata,
                )
            )
        return RunShieldResponse(violation=None)
