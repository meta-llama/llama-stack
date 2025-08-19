# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.datasetio,
            provider_type="inline::localfs",
            module="llama_stack.providers.inline.datasetio.localfs",
            config_class="llama_stack.providers.inline.datasetio.localfs.LocalFSDatasetIOConfig",
            api_dependencies=[],
            description="Local filesystem-based dataset I/O provider for reading and writing datasets to local storage.",
        ),
        remote_provider_spec(
            api=Api.datasetio,
            adapter=AdapterSpec(
                adapter_type="huggingface",
                pip_packages=[
                    "datasets",
                ],
                module="llama_stack.providers.remote.datasetio.huggingface",
                config_class="llama_stack.providers.remote.datasetio.huggingface.HuggingfaceDatasetIOConfig",
                description="HuggingFace datasets provider for accessing and managing datasets from the HuggingFace Hub.",
            ),
        ),
        remote_provider_spec(
            api=Api.datasetio,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                pip_packages=[
                    "datasets",
                ],
                module="llama_stack.providers.remote.datasetio.nvidia",
                config_class="llama_stack.providers.remote.datasetio.nvidia.NvidiaDatasetIOConfig",
                description="NVIDIA's dataset I/O provider for accessing datasets from NVIDIA's data platform.",
            ),
        ),
    ]
