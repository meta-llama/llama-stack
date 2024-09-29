# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_models.sku_list import resolve_model
from together import Together

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.safety import (
    RunShieldResponse,
    Safety,
    SafetyViolation,
    ViolationLevel,
)
from llama_stack.distribution.request_headers import NeedsRequestProviderData

from .config import TogetherSafetyConfig

SAFETY_SHIELD_TYPES = {
    "Llama-Guard-3-8B": "meta-llama/Meta-Llama-Guard-3-8B",
    "Llama-Guard-3-11B-Vision": "meta-llama/Llama-Guard-3-11B-Vision-Turbo",
}


def shield_type_to_model_name(shield_type: str) -> str:
    if shield_type == "llama_guard":
        shield_type = "Llama-Guard-3-8B"

    model = resolve_model(shield_type)
    if (
        model is None
        or not model.descriptor(shorten_default_variant=True) in SAFETY_SHIELD_TYPES
        or model.model_family is not ModelFamily.safety
    ):
        raise ValueError(
            f"{shield_type} is not supported, please use of {','.join(SAFETY_SHIELD_TYPES.keys())}"
        )

    return SAFETY_SHIELD_TYPES.get(model.descriptor(shorten_default_variant=True))


class TogetherSafetyImpl(Safety, NeedsRequestProviderData):
    def __init__(self, config: TogetherSafetyConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def run_shield(
        self, shield_type: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:

        together_api_key = None
        provider_data = self.get_request_provider_data()
        if provider_data is None or not provider_data.together_api_key:
            raise ValueError(
                'Pass Together API Key in the header X-LlamaStack-ProviderData as { "together_api_key": <your api key>}'
            )
        together_api_key = provider_data.together_api_key

        model_name = shield_type_to_model_name(shield_type)

        # messages can have role assistant or user
        api_messages = []
        for message in messages:
            if message.role in (Role.user.value, Role.assistant.value):
                api_messages.append({"role": message.role, "content": message.content})

        violation = await get_safety_response(
            together_api_key, model_name, api_messages
        )
        return RunShieldResponse(violation=violation)


async def get_safety_response(
    api_key: str, model_name: str, messages: List[Dict[str, str]]
) -> Optional[SafetyViolation]:
    client = Together(api_key=api_key)
    response = client.chat.completions.create(messages=messages, model=model_name)
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
