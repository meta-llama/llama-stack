# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List

from llama_stack.distribution.utils.model_utils import model_local_dir
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.distribution.datatypes import Api

from llama_stack.providers.datatypes import ShieldsProtocolPrivate

from .base import OnViolationAction, ShieldBase
from .config import SafetyConfig
from .llama_guard import LlamaGuardShield
from .prompt_guard import InjectionShield, JailbreakShield, PromptGuardShield


PROMPT_GUARD_MODEL = "Prompt-Guard-86M"


class MetaReferenceSafetyImpl(Safety, ShieldsProtocolPrivate):
    def __init__(self, config: SafetyConfig, deps) -> None:
        self.config = config
        self.inference_api = deps[Api.inference]

        self.available_shields = []
        if config.llama_guard_shield:
            self.available_shields.append(ShieldType.llama_guard)
        if config.enable_prompt_guard:
            self.available_shields.append(ShieldType.prompt_guard)

    async def initialize(self) -> None:
        if self.config.enable_prompt_guard:
            model_dir = model_local_dir(PROMPT_GUARD_MODEL)
            _ = PromptGuardShield.instance(model_dir)

    async def shutdown(self) -> None:
        pass

    async def register_shield(self, shield: Shield) -> None:
        if shield.shield_type not in self.available_shields:
            raise ValueError(f"Shield type {shield.shield_type} not supported")

    async def run_shield(
        self,
        shield_id: str,
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
        shield = await self.shield_store.get_shield(shield_id)
        if not shield:
            raise ValueError(f"Shield {shield_id} not found")

        shield_impl = self.get_shield_impl(shield)

        messages = messages.copy()
        # some shields like llama-guard require the first message to be a user message
        # since this might be a tool call, first role might not be user
        if len(messages) > 0 and messages[0].role != Role.user.value:
            messages[0] = UserMessage(content=messages[0].content)

        # TODO: we can refactor ShieldBase, etc. to be inline with the API types
        res = await shield_impl.run(messages)
        violation = None
        if (
            res.is_violation
            and shield_impl.on_violation_action != OnViolationAction.IGNORE
        ):
            violation = SafetyViolation(
                violation_level=(
                    ViolationLevel.ERROR
                    if shield_impl.on_violation_action == OnViolationAction.RAISE
                    else ViolationLevel.WARN
                ),
                user_message=res.violation_return_message,
                metadata={
                    "violation_type": res.violation_type,
                },
            )

        return RunShieldResponse(violation=violation)

    def get_shield_impl(self, shield: Shield) -> ShieldBase:
        if shield.shield_type == ShieldType.llama_guard:
            cfg = self.config.llama_guard_shield
            return LlamaGuardShield(
                model=cfg.model,
                inference_api=self.inference_api,
                excluded_categories=cfg.excluded_categories,
            )
        elif shield.shield_type == ShieldType.prompt_guard:
            model_dir = model_local_dir(PROMPT_GUARD_MODEL)
            subtype = shield.params.get("prompt_guard_type", "injection")
            if subtype == "injection":
                return InjectionShield.instance(model_dir)
            elif subtype == "jailbreak":
                return JailbreakShield.instance(model_dir)
            else:
                raise ValueError(f"Unknown prompt guard type: {subtype}")
        else:
            raise ValueError(f"Unknown shield type: {shield.shield_type}")
