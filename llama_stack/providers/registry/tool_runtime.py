# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.tool_runtime,
            provider_type="inline::brave-search",
            pip_packages=[],
            module="llama_stack.providers.inline.tool_runtime.brave_search",
            config_class="llama_stack.providers.inline.tool_runtime.brave_search.config.BraveSearchToolConfig",
            provider_data_validator="llama_stack.providers.inline.tool_runtime.brave_search.BraveSearchToolProviderDataValidator",
        ),
    ]
