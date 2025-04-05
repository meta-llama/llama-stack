# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import (
    Api,
    InlineProviderSpec,
    ProviderSpec,
)


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.preprocessing,
            provider_type="inline::basic",
            pip_packages=["httpx", "pypdf"],
            module="llama_stack.providers.inline.preprocessing.basic",
            config_class="llama_stack.providers.inline.preprocessing.basic.InlineBasicPreprocessorConfig",
            api_dependencies=[],
        ),
        InlineProviderSpec(
            api=Api.preprocessing,
            provider_type="inline::simple_chunking",
            pip_packages=[],
            module="llama_stack.providers.inline.preprocessing.simple_chunking",
            config_class="llama_stack.providers.inline.preprocessing.simple_chunking.InclineSimpleChunkingConfig",
            api_dependencies=[],
        ),
    ]
