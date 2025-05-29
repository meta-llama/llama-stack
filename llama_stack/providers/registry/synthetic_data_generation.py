# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.synthetic_data_generation,
            provider_type="inline::synthetic_data_kit",
            pip_packages=[
                "synthetic-data-kit",
                "vllm",
                "pydantic",
            ],
            module="llama_stack.providers.inline.synthetic_data_generation.synthetic_data_kit_inline",
            config_class="llama_stack.providers.inline.synthetic_data_generation.config.SyntheticDataKitConfig",
        ),
    ]
