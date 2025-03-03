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
            provider_type="inline::docling",
            pip_packages=["docling"],
            module="llama_stack.providers.inline.preprocessing.docling",
            config_class="llama_stack.providers.inline.preprocessing.docling.InlineDoclingConfig",
            api_dependencies=[],
        ),
    ]
