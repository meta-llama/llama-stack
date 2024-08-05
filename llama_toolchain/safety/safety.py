# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from typing import Dict

from llama_toolchain.distribution.datatypes import Api, ProviderSpec

from .config import SafetyConfig
from .api.endpoints import *  # noqa
from .shields import (
    CodeScannerShield,
    InjectionShield,
    JailbreakShield,
    LlamaGuardShield,
    PromptGuardShield,
    ShieldBase,
    ThirdPartyShield,
)


async def get_provider_impl(config: SafetyConfig, _deps: Dict[Api, ProviderSpec]):
    assert isinstance(config, SafetyConfig), f"Unexpected config type: {type(config)}"

    impl = MetaReferenceSafetyImpl(config)
    await impl.initialize()
    return impl


class MetaReferenceSafetyImpl(Safety):

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        shield_cfg = self.config.llama_guard_shield
        if shield_cfg is not None:
            _ = LlamaGuardShield.instance(
                model_dir=shield_cfg.model_dir,
                excluded_categories=shield_cfg.excluded_categories,
                disable_input_check=shield_cfg.disable_input_check,
                disable_output_check=shield_cfg.disable_output_check,
            )

        shield_cfg = self.config.prompt_guard_shield
        if shield_cfg is not None:
            _ = PromptGuardShield.instance(shield_cfg.model_dir)

    async def run_shields(
        self,
        request: RunShieldRequest,
    ) -> RunShieldResponse:
        shields = [shield_config_to_shield(c, self.config) for c in request.shields]

        responses = await asyncio.gather(
            *[shield.run(request.messages) for shield in shields]
        )

        return RunShieldResponse(responses=responses)


def shield_config_to_shield(
    sc: ShieldDefinition, safety_config: SafetyConfig
) -> ShieldBase:
    if sc.shield_type == BuiltinShield.llama_guard:
        assert (
            safety_config.llama_guard_shield is not None
        ), "Cannot use LlamaGuardShield since not present in config"
        return LlamaGuardShield.instance(
            model_dir=safety_config.llama_guard_shield.model_dir
        )
    elif sc.shield_type == BuiltinShield.jailbreak_shield:
        assert (
            safety_config.prompt_guard_shield is not None
        ), "Cannot use Jailbreak Shield since Prompt Guard not present in config"
        return JailbreakShield.instance(safety_config.prompt_guard_shield.model_dir)
    elif sc.shield_type == BuiltinShield.injection_shield:
        assert (
            safety_config.prompt_guard_shield is not None
        ), "Cannot use PromptGuardShield since not present in config"
        return InjectionShield.instance(safety_config.prompt_guard_shield.model_dir)
    elif sc.shield_type == BuiltinShield.code_scanner_guard:
        return CodeScannerShield.instance()
    elif sc.shield_type == BuiltinShield.third_party_shield:
        return ThirdPartyShield.instance()
    else:
        raise ValueError(f"Unknown shield type: {sc.shield_type}")
