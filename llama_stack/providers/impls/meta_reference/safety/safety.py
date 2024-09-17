# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from llama_models.sku_list import resolve_model

from llama_stack.distribution.utils.model_utils import model_local_dir
from llama_stack.apis.safety import *  # noqa

from .config import SafetyConfig
from .shields import (
    CodeScannerShield,
    InjectionShield,
    JailbreakShield,
    LlamaGuardShield,
    PromptGuardShield,
    ShieldBase,
    ThirdPartyShield,
)


def resolve_and_get_path(model_name: str) -> str:
    model = resolve_model(model_name)
    assert model is not None, f"Could not resolve model {model_name}"
    model_dir = model_local_dir(model.descriptor())
    return model_dir


class MetaReferenceSafetyImpl(Safety):
    def __init__(self, config: SafetyConfig) -> None:
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

    async def run_shields(
        self,
        messages: List[Message],
        shields: List[ShieldDefinition],
    ) -> RunShieldResponse:
        shields = [shield_config_to_shield(c, self.config) for c in shields]

        responses = await asyncio.gather(*[shield.run(messages) for shield in shields])

        return RunShieldResponse(responses=responses)


def shield_type_equals(a: ShieldType, b: ShieldType):
    return a == b or a == b.value


def shield_config_to_shield(
    sc: ShieldDefinition, safety_config: SafetyConfig
) -> ShieldBase:
    if shield_type_equals(sc.shield_type, BuiltinShield.llama_guard):
        assert (
            safety_config.llama_guard_shield is not None
        ), "Cannot use LlamaGuardShield since not present in config"
        model_dir = resolve_and_get_path(safety_config.llama_guard_shield.model)
        return LlamaGuardShield.instance(model_dir=model_dir)
    elif shield_type_equals(sc.shield_type, BuiltinShield.jailbreak_shield):
        assert (
            safety_config.prompt_guard_shield is not None
        ), "Cannot use Jailbreak Shield since Prompt Guard not present in config"
        model_dir = resolve_and_get_path(safety_config.prompt_guard_shield.model)
        return JailbreakShield.instance(model_dir)
    elif shield_type_equals(sc.shield_type, BuiltinShield.injection_shield):
        assert (
            safety_config.prompt_guard_shield is not None
        ), "Cannot use PromptGuardShield since not present in config"
        model_dir = resolve_and_get_path(safety_config.prompt_guard_shield.model)
        return InjectionShield.instance(model_dir)
    elif shield_type_equals(sc.shield_type, BuiltinShield.code_scanner_guard):
        return CodeScannerShield.instance()
    elif shield_type_equals(sc.shield_type, BuiltinShield.third_party_shield):
        return ThirdPartyShield.instance()
    else:
        raise ValueError(f"Unknown shield type: {sc.shield_type}")
