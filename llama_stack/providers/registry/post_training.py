# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import AdapterSpec, Api, InlineProviderSpec, ProviderSpec, remote_provider_spec


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.post_training,
            provider_type="inline::torchtune",
            module="llama_stack.providers.inline.post_training.torchtune",
            config_class="llama_stack.providers.inline.post_training.torchtune.TorchtunePostTrainingConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
            description="TorchTune-based post-training provider for fine-tuning and optimizing models using Meta's TorchTune framework.",
        ),
        InlineProviderSpec(
            api=Api.post_training,
            provider_type="inline::huggingface",
            module="llama_stack.providers.inline.post_training.huggingface",
            config_class="llama_stack.providers.inline.post_training.huggingface.HuggingFacePostTrainingConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
            ],
            description="HuggingFace-based post-training provider for fine-tuning models using the HuggingFace ecosystem.",
        ),
        remote_provider_spec(
            api=Api.post_training,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                module="llama_stack.providers.remote.post_training.nvidia",
                config_class="llama_stack.providers.remote.post_training.nvidia.NvidiaPostTrainingConfig",
                description="NVIDIA's post-training provider for fine-tuning models on NVIDIA's platform.",
            ),
        ),
    ]
