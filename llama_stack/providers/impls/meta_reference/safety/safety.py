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
from llama_stack.distribution.datatypes import Api, RoutableProvider

from llama_stack.providers.impls.meta_reference.safety.shields.base import (
    OnViolationAction,
)

from .config import MetaReferenceShieldType, SafetyConfig

from .shields import CodeScannerShield, LlamaGuardShield, ShieldBase

PROMPT_GUARD_MODEL = "Prompt-Guard-86M"


class MetaReferenceSafetyImpl(Safety, RoutableProvider):
    def __init__(self, config: SafetyConfig, deps) -> None:
        self.config = config
        self.inference_api = deps[Api.inference]

    async def initialize(self) -> None:
        if self.config.enable_prompt_guard:
            from .shields import PromptGuardShield

            model_dir = model_local_dir(PROMPT_GUARD_MODEL)
            _ = PromptGuardShield.instance(model_dir)

    async def shutdown(self) -> None:
        pass

    async def validate_routing_keys(self, routing_keys: List[str]) -> None:
        available_shields = [v.value for v in MetaReferenceShieldType]
        for key in routing_keys:
            if key not in available_shields:
                raise ValueError(f"Unknown safety shield type: {key}")

    async def run_shield(
        self,
        shield_type: str,
        messages: List[Message],
        params: Dict[str, Any] = None,
    ) -> RunShieldResponse:
        available_shields = [v.value for v in MetaReferenceShieldType]
        assert shield_type in available_shields, f"Unknown shield {shield_type}"

        shield = self.get_shield_impl(MetaReferenceShieldType(shield_type))

        messages = messages.copy()
        # some shields like llama-guard require the first message to be a user message
        # since this might be a tool call, first role might not be user
        if len(messages) > 0 and messages[0].role != Role.user.value:
            messages[0] = UserMessage(content=messages[0].content)

        # TODO: we can refactor ShieldBase, etc. to be inline with the API types
        res = await shield.run(messages)
        violation = None
        if res.is_violation and shield.on_violation_action != OnViolationAction.IGNORE:
            violation = SafetyViolation(
                violation_level=(
                    ViolationLevel.ERROR
                    if shield.on_violation_action == OnViolationAction.RAISE
                    else ViolationLevel.WARN
                ),
                user_message=res.violation_return_message,
                metadata={
                    "violation_type": res.violation_type,
                },
            )

        return RunShieldResponse(violation=violation)

    def get_shield_impl(self, typ: MetaReferenceShieldType) -> ShieldBase:
        cfg = self.config
        if typ == MetaReferenceShieldType.llama_guard:
            cfg = cfg.llama_guard_shield
            assert (
                cfg is not None
            ), "Cannot use LlamaGuardShield since not present in config"

            return LlamaGuardShield(
                model=cfg.model,
                inference_api=self.inference_api,
                excluded_categories=cfg.excluded_categories,
                disable_input_check=cfg.disable_input_check,
                disable_output_check=cfg.disable_output_check,
            )
        elif typ == MetaReferenceShieldType.jailbreak_shield:
            from .shields import JailbreakShield

            model_dir = model_local_dir(PROMPT_GUARD_MODEL)
            return JailbreakShield.instance(model_dir)
        elif typ == MetaReferenceShieldType.injection_shield:
            from .shields import InjectionShield

            model_dir = model_local_dir(PROMPT_GUARD_MODEL)
            return InjectionShield.instance(model_dir)
        elif typ == MetaReferenceShieldType.code_scanner_guard:
            return CodeScannerShield.instance()
        else:
            raise ValueError(f"Unknown shield type: {typ}")
