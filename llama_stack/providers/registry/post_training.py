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
            api=Api.post_training,
            provider_type="inline::torchtune",
            pip_packages=["torch", "torchtune", "torchao", "numpy"],
            module="llama_stack.providers.inline.post_training.torchtune",
            config_class="llama_stack.providers.inline.post_training.torchtune.TorchtunePostTrainingConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
        ),
    ]
