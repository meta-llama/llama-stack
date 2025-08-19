# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.providers.datatypes import AdapterSpec, Api, InlineProviderSpec, ProviderSpec, remote_provider_spec


def available_providers() -> list[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.eval,
            provider_type="inline::meta-reference",
            module="llama_stack.providers.inline.eval.meta_reference",
            config_class="llama_stack.providers.inline.eval.meta_reference.MetaReferenceEvalConfig",
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
                Api.scoring,
                Api.inference,
                Api.agents,
            ],
            description="Meta's reference implementation of evaluation tasks with support for multiple languages and evaluation metrics.",
        ),
        remote_provider_spec(
            api=Api.eval,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                pip_packages=[
                    "requests",
                ],
                module="llama_stack.providers.remote.eval.nvidia",
                config_class="llama_stack.providers.remote.eval.nvidia.NVIDIAEvalConfig",
                description="NVIDIA's evaluation provider for running evaluation tasks on NVIDIA's platform.",
            ),
            api_dependencies=[
                Api.datasetio,
                Api.datasets,
                Api.scoring,
                Api.inference,
                Api.agents,
            ],
        ),
    ]
