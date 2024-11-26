# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.distribution.datatypes import *  # noqa: F403


META_REFERENCE_DEPS = [
    "accelerate",
    "blobfile",
    "fairscale",
    "torch",
    "torchvision",
    "transformers",
    "zmq",
    "lm-format-enforcer",
]


def available_providers() -> List[ProviderSpec]:
    return [
        InlineProviderSpec(
            api=Api.inference,
            provider_type="inline::meta-reference",
            pip_packages=META_REFERENCE_DEPS,
            module="llama_stack.providers.inline.inference.meta_reference",
            config_class="llama_stack.providers.inline.inference.meta_reference.MetaReferenceInferenceConfig",
        ),
        InlineProviderSpec(
            api=Api.inference,
            provider_type="inline::meta-reference-quantized",
            pip_packages=(
                META_REFERENCE_DEPS
                + [
                    "fbgemm-gpu",
                    "torchao==0.5.0",
                ]
            ),
            module="llama_stack.providers.inline.inference.meta_reference",
            config_class="llama_stack.providers.inline.inference.meta_reference.MetaReferenceQuantizedInferenceConfig",
        ),
        InlineProviderSpec(
            api=Api.inference,
            provider_type="inline::vllm",
            pip_packages=[
                "vllm",
            ],
            module="llama_stack.providers.inline.inference.vllm",
            config_class="llama_stack.providers.inline.inference.vllm.VLLMConfig",
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="sample",
                pip_packages=[],
                module="llama_stack.providers.remote.inference.sample",
                config_class="llama_stack.providers.remote.inference.sample.SampleConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="ollama",
                pip_packages=["ollama", "aiohttp"],
                config_class="llama_stack.providers.remote.inference.ollama.OllamaImplConfig",
                module="llama_stack.providers.remote.inference.ollama",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="vllm",
                pip_packages=["openai"],
                module="llama_stack.providers.remote.inference.vllm",
                config_class="llama_stack.providers.remote.inference.vllm.VLLMInferenceAdapterConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="tgi",
                pip_packages=["huggingface_hub", "aiohttp"],
                module="llama_stack.providers.remote.inference.tgi",
                config_class="llama_stack.providers.remote.inference.tgi.TGIImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="hf::serverless",
                pip_packages=["huggingface_hub", "aiohttp"],
                module="llama_stack.providers.remote.inference.tgi",
                config_class="llama_stack.providers.remote.inference.tgi.InferenceAPIImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="hf::endpoint",
                pip_packages=["huggingface_hub", "aiohttp"],
                module="llama_stack.providers.remote.inference.tgi",
                config_class="llama_stack.providers.remote.inference.tgi.InferenceEndpointImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="fireworks",
                pip_packages=[
                    "fireworks-ai",
                ],
                module="llama_stack.providers.remote.inference.fireworks",
                config_class="llama_stack.providers.remote.inference.fireworks.FireworksImplConfig",
                provider_data_validator="llama_stack.providers.remote.inference.fireworks.FireworksProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="together",
                pip_packages=[
                    "together",
                ],
                module="llama_stack.providers.remote.inference.together",
                config_class="llama_stack.providers.remote.inference.together.TogetherImplConfig",
                provider_data_validator="llama_stack.providers.remote.inference.together.TogetherProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="bedrock",
                pip_packages=["boto3"],
                module="llama_stack.providers.remote.inference.bedrock",
                config_class="llama_stack.providers.remote.inference.bedrock.BedrockConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="databricks",
                pip_packages=[
                    "openai",
                ],
                module="llama_stack.providers.remote.inference.databricks",
                config_class="llama_stack.providers.remote.inference.databricks.DatabricksImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="clarifai",
                pip_packages=[
                    "clarifai",
                ],
                module="llama_stack.providers.adapters.inference.clarifai",
                config_class="llama_stack.providers.adapters.inference.clarifai.ClarifaiImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="nvidia",
                pip_packages=[
                    "openai",
                ],
                module="llama_stack.providers.remote.inference.nvidia",
                config_class="llama_stack.providers.remote.inference.nvidia.NVIDIAConfig",
            ),
        ),
    ]
