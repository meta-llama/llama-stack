# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import Api, InlineProviderSpec, ProviderSpec


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.post_training,
            provider_type="inline::torchtune",
            pip_packages=["torch", "torchtune==0.5.0", "torchao==0.8.0", "numpy", "openai"],
            module="llama_stack.providers.inline.post_training.torchtune",
            config_class="llama_stack.providers.inline.post_training.torchtune.TorchtunePostTrainingConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
        ),
        InlineProviderSpec(
            api=Api.post_training,
            provider_type="inline::huggingface-ilab",
            pip_packages=["torch", "transformers", "datasets", "numpy", "openai"],
            module="llama_stack.providers.inline.post_training.huggingface_ilab",
            config_class="llama_stack.providers.inline.post_training.huggingface_ilab.HFilabPostTrainingConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
        ),
    ]
