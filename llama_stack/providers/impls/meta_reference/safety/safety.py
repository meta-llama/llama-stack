# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.sku_list import resolve_model

from llama_stack.distribution.utils.model_utils import model_local_dir
from llama_stack.apis.safety import *  # noqa: F403
from llama_models.llama3.api.datatypes import *  # noqa: F403

from .config import MetaReferenceShieldType, SafetyConfig

from .shields import (
    CodeScannerShield,
    InjectionShield,
    JailbreakShield,
    LlamaGuardShield,
    PromptGuardShield,
    ShieldBase,
)


def resolve_and_get_path(model_name: str) -> str:
    model = resolve_model(model_name)
    assert model is not None, f"Could not resolve model {model_name}"
    model_dir = model_local_dir(model.descriptor())
    return model_dir


class MetaReferenceSafetyImpl(Safety):
    def __init__(self, config: SafetyConfig) -> None:
        print("Initializing MetaReferenceSafetyImpl w/ config", config)
        self.config = config

    async def initialize(self) -> None:
        shield_cfg = self.config.llama_guard_shield
        if shield_cfg is not None:
            model_dir = resolve_and_get_path(shield_cfg.model)
            _ = LlamaGuardShield.instance(
                model_dir=model_dir,
                excluded_categories=shield_cfg.excluded_categories,
                disable_input_check=shield_cfg.disable_input_check,
                disable_output_check=shield_cfg.disable_output_check,
            )

        shield_cfg = self.config.prompt_guard_shield
        if shield_cfg is not None:
            model_dir = resolve_and_get_path(shield_cfg.model)
            _ = PromptGuardShield.instance(model_dir)

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
        if res.is_violation:
            violation = SafetyViolation(
                violation_level=ViolationLevel.ERROR,
                user_message=res.violation_return_message,
                metadata={
                    "violation_type": res.violation_type,
                },
            )

        return RunShieldResponse(violation=violation)

    def get_shield_impl(self, typ: MetaReferenceShieldType) -> ShieldBase:
        cfg = self.config
        if typ == MetaReferenceShieldType.llama_guard:
            assert (
                cfg.llama_guard_shield is not None
            ), "Cannot use LlamaGuardShield since not present in config"
            model_dir = resolve_and_get_path(cfg.llama_guard_shield.model)
            return LlamaGuardShield.instance(model_dir=model_dir)
        elif typ == MetaReferenceShieldType.jailbreak_shield:
            assert (
                cfg.prompt_guard_shield is not None
            ), "Cannot use Jailbreak Shield since Prompt Guard not present in config"
            model_dir = resolve_and_get_path(cfg.prompt_guard_shield.model)
            return JailbreakShield.instance(model_dir)
        elif typ == MetaReferenceShieldType.injection_shield:
            assert (
                cfg.prompt_guard_shield is not None
            ), "Cannot use PromptGuardShield since not present in config"
            model_dir = resolve_and_get_path(cfg.prompt_guard_shield.model)
            return InjectionShield.instance(model_dir)
        elif typ == MetaReferenceShieldType.code_scanner_guard:
            return CodeScannerShield.instance()
        else:
            raise ValueError(f"Unknown shield type: {typ}")
