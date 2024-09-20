# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.safety,
            provider_id="meta-reference",
            pip_packages=[
                "accelerate",
                "codeshield",
                "torch",
                "transformers",
            ],
            module="llama_stack.providers.impls.meta_reference.safety",
            config_class="llama_stack.providers.impls.meta_reference.safety.SafetyConfig",
        ),
        remote_provider_spec(
            api=Api.safety,
            adapter=AdapterSpec(
                adapter_id="bedrock",
                pip_packages=[
                    "aws-sdk",
                ],
                module="llama_stack.providers.adapters.safety.bedrock",
                header_extractor="llama_stack.providers.adapters.safety.bedrock.BedrockSafetyRequestProviderData",
            ),
        ),
    ]
