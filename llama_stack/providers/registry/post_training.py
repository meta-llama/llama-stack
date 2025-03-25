# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import AdapterSpec, Api, InlineProviderSpec, ProviderSpec, remote_provider_spec


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.post_training,
            provider_type="inline::torchtune",
            pip_packages=["torch", "torchtune==0.5.0", "torchao==0.8.0", "numpy"],
            module="llama_stack.providers.inline.post_training.torchtune",
            config_class="llama_stack.providers.inline.post_training.torchtune.TorchtunePostTrainingConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
        ),
        remote_provider_spec(
            api=Api.post_training,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                pip_packages=["requests", "aiohttp"],
                module="llama_stack.providers.remote.post_training.nvidia",
                config_class="llama_stack.providers.remote.post_training.nvidia.NvidiaPostTrainingConfig",
            ),
        ),
    ]
