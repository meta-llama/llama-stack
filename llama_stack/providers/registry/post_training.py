# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import *  # noqa: F403


META_REFERENCE_DEPS = [
    "torch",
    "torchtune",
    "torchao",
    "numpy",
]


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.post_training,
            provider_type="inline::meta-reference",
            pip_packages=META_REFERENCE_DEPS,
            module="llama_stack.providers.inline.post_training.meta_reference",
            config_class="llama_stack.providers.inline.post_training.meta_reference.MetaReferencePostTrainingConfig",
            api_dependencies=[
                Api.datasetio,
            ],
        ),
    ]
