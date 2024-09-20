# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from together import Together

from llama_stack.distribution.request_headers import get_request_provider_data

from .config import TogetherProviderDataValidator, TogetherSafetyConfig


class TogetherSafetyImpl(Safety):
    def __init__(self, config: TogetherSafetyConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def run_shield(
        self, shield_type: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        if shield_type != "llama_guard":
            raise ValueError(f"shield type {shield_type} is not supported")

        provider_data = get_request_provider_data()

        together_api_key = None
        if provider_data is not None:
            if not isinstance(provider_data, TogetherProviderDataValidator):
                raise ValueError(
                    'Pass Together API Key in the header X-LlamaStack-ProviderData as { "together_api_key": <your api key>}'
                )

            together_api_key = provider_data.together_api_key
        if not together_api_key:
            together_api_key = self.config.api_key

        if not together_api_key:
            raise ValueError("The API key must be provider in the header or config")

        # messages can have role assistant or user
        api_messages = []
        for message in messages:
            if message.role in (Role.user.value, Role.assistant.value):
                api_messages.append({"role": message.role, "content": message.content})

        violation = await get_safety_response(together_api_key, api_messages)
        return RunShieldResponse(violation=violation)


async def get_safety_response(
    api_key: str, messages: List[Dict[str, str]]
) -> Optional[SafetyViolation]:
    client = Together(api_key=api_key)
    response = client.chat.completions.create(
        messages=messages, model="meta-llama/Meta-Llama-Guard-3-8B"
    )
    if len(response.choices) == 0:
        return None

    response_text = response.choices[0].message.content
    if response_text == "safe":
        return None

    parts = response_text.split("\n")
    if len(parts) != 2:
        return None

    if parts[0] == "unsafe":
        return SafetyViolation(
            violation_level=ViolationLevel.ERROR,
            user_message="unsafe",
            metadata={"violation_type": parts[1]},
        )

    return None
