# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.safety,
            provider_type="inline::prompt-guard",
            pip_packages=[
                "transformers[accelerate]",
                "torch --index-url https://download.pytorch.org/whl/cpu",
            ],
            module="llama_stack.providers.inline.safety.prompt_guard",
            config_class="llama_stack.providers.inline.safety.prompt_guard.PromptGuardConfig",
        ),
        InlineProviderSpec(
            api=Api.safety,
            provider_type="inline::llama-guard",
            pip_packages=[],
            module="llama_stack.providers.inline.safety.llama_guard",
            config_class="llama_stack.providers.inline.safety.llama_guard.LlamaGuardConfig",
            api_dependencies=[
                Api.inference,
            ],
        ),
        InlineProviderSpec(
            api=Api.safety,
            provider_type="inline::code-scanner",
            pip_packages=[
                "codeshield",
            ],
            module="llama_stack.providers.inline.safety.code_scanner",
            config_class="llama_stack.providers.inline.safety.code_scanner.CodeScannerConfig",
        ),
        remote_provider_spec(
            api=Api.safety,
            adapter=AdapterSpec(
                adapter_type="bedrock",
                pip_packages=["boto3"],
                module="llama_stack.providers.remote.safety.bedrock",
                config_class="llama_stack.providers.remote.safety.bedrock.BedrockSafetyConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.safety,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                pip_packages=["requests"],
                module="llama_stack.providers.remote.safety.nvidia",
                config_class="llama_stack.providers.remote.safety.nvidia.NVIDIASafetyConfig",
            ),
        ),
    ]
