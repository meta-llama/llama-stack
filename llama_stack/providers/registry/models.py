# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import *  # noqa: F403


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.models,
            provider_id="builtin",
            pip_packages=[],
            module="llama_stack.providers.impls.builtin.models",
            config_class="llama_stack.providers.impls.builtin.models.BuiltinImplConfig",
            api_dependencies=[],
        )
    ]
