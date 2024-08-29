# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_toolchain.core.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_safety_providers() -> List[ProviderSpec]:
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
            module="llama_toolchain.safety.meta_reference",
            config_class="llama_toolchain.safety.meta_reference.SafetyConfig",
        ),
    ]
