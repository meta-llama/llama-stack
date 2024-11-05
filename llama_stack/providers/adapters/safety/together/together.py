# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from together import Together

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import ShieldsProtocolPrivate

from .config import TogetherSafetyConfig


TOGETHER_SHIELD_MODEL_MAP = {
    "llama_guard": "meta-llama/Meta-Llama-Guard-3-8B",
    "Llama-Guard-3-8B": "meta-llama/Meta-Llama-Guard-3-8B",
    "Llama-Guard-3-11B-Vision": "meta-llama/Llama-Guard-3-11B-Vision-Turbo",
}


class TogetherSafetyImpl(Safety, NeedsRequestProviderData, ShieldsProtocolPrivate):
    def __init__(self, config: TogetherSafetyConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: ShieldDef) -> None:
        raise ValueError("Registering dynamic shields is not supported")

    async def list_shields(self) -> List[ShieldDef]:
        return [
            ShieldDef(
                identifier=ShieldType.llama_guard.value,
                shield_type=ShieldType.llama_guard.value,
                params={},
            )
        ]

    async def run_shield(
        self, shield_type: str, messages: List[Message], params: Dict[str, Any] = None
    ) -> RunShieldResponse:
        shield_def = await self.shield_store.get_shield(shield_type)
        if not shield_def:
            raise ValueError(f"Unknown shield {shield_type}")

        model = shield_def.params.get("model", "llama_guard")
        if model not in TOGETHER_SHIELD_MODEL_MAP:
            raise ValueError(f"Unsupported safety model: {model}")

        together_api_key = None
        if self.config.api_key is not None:
            together_api_key = self.config.api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.together_api_key:
                raise ValueError(
                    'Pass Together API Key in the header X-LlamaStack-ProviderData as { "together_api_key": <your api key>}'
                )
            together_api_key = provider_data.together_api_key

        # messages can have role assistant or user
        api_messages = []
        for message in messages:
            if message.role in (Role.user.value, Role.assistant.value):
                api_messages.append({"role": message.role, "content": message.content})

        violation = await get_safety_response(
            together_api_key, TOGETHER_SHIELD_MODEL_MAP[model], api_messages
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
            metadata={"violation_type": parts[1]},
        )

    return None
