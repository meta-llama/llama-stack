# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import (
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
            provider_type="meta-reference",
            pip_packages=[
                "transformers",
                "torch --index-url https://download.pytorch.org/whl/cpu",
            ],
            module="llama_stack.providers.inline.safety.meta_reference",
            config_class="llama_stack.providers.inline.safety.meta_reference.SafetyConfig",
            api_dependencies=[
                Api.inference,
            ],
        ),
        remote_provider_spec(
            api=Api.safety,
            adapter=AdapterSpec(
                adapter_type="sample",
                pip_packages=[],
                module="llama_stack.providers.remote.safety.sample",
                config_class="llama_stack.providers.remote.safety.sample.SampleConfig",
            ),
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
        InlineProviderSpec(
            api=Api.safety,
            provider_type="meta-reference/codeshield",
            pip_packages=[
                "codeshield",
            ],
            module="llama_stack.providers.inline.safety.meta_reference",
            config_class="llama_stack.providers.inline.safety.meta_reference.CodeShieldConfig",
            api_dependencies=[],
        ),
    ]
