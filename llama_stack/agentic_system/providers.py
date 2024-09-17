# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.core.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.agentic_system,
            provider_id="meta-reference",
            pip_packages=[
                "codeshield",
                "matplotlib",
                "pillow",
                "pandas",
                "scikit-learn",
                "torch",
                "transformers",
            ],
            module="llama_stack.agentic_system.meta_reference",
            config_class="llama_stack.agentic_system.meta_reference.MetaReferenceImplConfig",
            api_dependencies=[
                Api.inference,
                Api.safety,
                Api.memory,
            ],
        ),
    ]
