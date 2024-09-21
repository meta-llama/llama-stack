# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import pydantic
from together import Together

import asyncio
from .config import TogetherSafetyConfig
from llama_stack.apis.safety import *
import logging


class TogetherSafetyImpl(Safety):
    def __init__(self, config: TogetherSafetyConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self) -> Together:
        if self._client == None:
            self._client =  Together(api_key=self.config.api_key)
        return self._client

    @client.setter
    def client(self, client: Together) -> None:
        self._client = client


    async def initialize(self) -> None:
        pass

    async def run_shields(
            self,
            messages: List[Message],
            shields: List[ShieldDefinition],
    ) -> RunShieldResponse:
        # support only llama guard shield
        for shield in shields:
            if not isinstance(shield.shield_type, BuiltinShield) or shield.shield_type != BuiltinShield.llama_guard:
                raise ValueError(f"shield type {shield.shield_type} is not supported")

        # messages can have role assistant or user
        api_messages = []
        for message in messages:
            if type(message) is UserMessage:
                api_messages.append({'role': message.role, 'content': message.content})
            else:
                raise ValueError(f"role {message.role} is not supported")

        # construct Together request
        responses = await asyncio.gather(*[get_safety_response(self.client, api_messages)])
        return RunShieldResponse(responses=responses)

async def get_safety_response(client: Together, messages: List[Dict[str, str]]) -> Optional[ShieldResponse]:
    response = client.chat.completions.create(messages=messages, model="meta-llama/Meta-Llama-Guard-3-8B")
    if len(response.choices) == 0:
        return ShieldResponse(shield_type=BuiltinShield.llama_guard, is_violation=False)

    response_text = response.choices[0].message.content
    if response_text == 'safe':
        return ShieldResponse(
            shield_type=BuiltinShield.llama_guard,
            is_violation=False,
        )
    else:
        parts = response_text.split("\n")
        if not len(parts) == 2:
            return None

        if parts[0] == 'unsafe':
            return ShieldResponse(
                shield_type=BuiltinShield.llama_guard,
                is_violation=True,
                violation_type=parts[1],
                violation_return_message="Sorry, I cannot do that"
            )

    return None




