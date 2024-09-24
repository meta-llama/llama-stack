# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import pydantic
from together import Together

import asyncio

from llama_stack.distribution.request_headers import get_request_provider_data
from .config import TogetherSafetyConfig
from llama_stack.apis.safety import *
import logging

class TogetherHeaderInfo(BaseModel):
    together_api_key: str


class TogetherSafetyImpl(Safety):
    def __init__(self, config: TogetherSafetyConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def run_shield(
            self,
            shield_type: str,
            messages: List[Message],
            params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
        # support only llama guard shield

        if shield_type != "llama_guard":
            raise ValueError(f"shield type {shield_type} is not supported")

        provider_data = get_request_provider_data()
        together_api_key = None
        # @TODO error out if together_api_key is missing in the header
        if provider_data is not None:
            if not isinstance(provider_data, TogetherHeaderInfo) or provider_data.together_api_key is None:
                raise ValueError("provider Together api key in the header X-LlamaStack-ProviderData as { \"together_api_key\": <your api key>}")

            together_api_key = provider_data.together_api_key

        # messages can have role assistant or user
        api_messages = []
        for message in messages:
            if type(message) is UserMessage:
                api_messages.append({'role': message.role, 'content': message.content})
            else:
                raise ValueError(f"role {message.role} is not supported")

        # construct Together request
        response = await asyncio.run(get_safety_response(together_api_key, api_messages))
        return RunShieldResponse(violation=response)

async def get_safety_response(api_key: str, messages: List[Dict[str, str]]) -> Optional[SafetyViolation]:
    client = Together(api_key=api_key)
    response = client.chat.completions.create(messages=messages, model="meta-llama/Meta-Llama-Guard-3-8B")
    if len(response.choices) == 0:
        return SafetyViolation(violation_level=ViolationLevel.INFO, user_message="safe")

    response_text = response.choices[0].message.content
    if response_text == 'safe':
        return SafetyViolation(violation_level=ViolationLevel.INFO, user_message="safe")
    else:
        parts = response_text.split("\n")
        if not len(parts) == 2:
            return None

        if parts[0] == 'unsafe':
            SafetyViolation(violation_level=ViolationLevel.WARN, user_message="unsafe",
                            metadata={"violation_type": parts[1]})

    return None




