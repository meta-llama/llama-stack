# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

from llama_stack.providers.datatypes import (
    AdapterSpec,
    Api,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)

META_REFERENCE_DEPS = [
    "accelerate",
    "blobfile",
    "fairscale",
    "torch",
    "torchvision",
    "transformers",
    "zmq",
    "lm-format-enforcer",
    "sentence-transformers",
    "torchao==0.8.0",
    "fbgemm-gpu-genai==1.1.2",
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
            provider_type="inline::vllm",
            pip_packages=[
                "vllm",
            ],
            module="llama_stack.providers.inline.inference.vllm",
            config_class="llama_stack.providers.inline.inference.vllm.VLLMConfig",
        ),
        InlineProviderSpec(
            api=Api.inference,
            provider_type="inline::sentence-transformers",
            pip_packages=[
                "torch torchvision --index-url https://download.pytorch.org/whl/cpu",
                "sentence-transformers --no-deps",
            ],
            module="llama_stack.providers.inline.inference.sentence_transformers",
            config_class="llama_stack.providers.inline.inference.sentence_transformers.config.SentenceTransformersInferenceConfig",
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="cerebras",
                pip_packages=[
                    "cerebras_cloud_sdk",
                ],
                module="llama_stack.providers.remote.inference.cerebras",
                config_class="llama_stack.providers.remote.inference.cerebras.CerebrasImplConfig",
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
                adapter_type="nvidia",
                pip_packages=[
                    "openai",
                ],
                module="llama_stack.providers.remote.inference.nvidia",
                config_class="llama_stack.providers.remote.inference.nvidia.NVIDIAConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="runpod",
                pip_packages=["openai"],
                module="llama_stack.providers.remote.inference.runpod",
                config_class="llama_stack.providers.remote.inference.runpod.RunpodImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="openai",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.openai",
                config_class="llama_stack.providers.remote.inference.openai.OpenAIConfig",
                provider_data_validator="llama_stack.providers.remote.inference.openai.config.OpenAIProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="anthropic",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.anthropic",
                config_class="llama_stack.providers.remote.inference.anthropic.AnthropicConfig",
                provider_data_validator="llama_stack.providers.remote.inference.anthropic.config.AnthropicProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="gemini",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.gemini",
                config_class="llama_stack.providers.remote.inference.gemini.GeminiConfig",
                provider_data_validator="llama_stack.providers.remote.inference.gemini.config.GeminiProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="groq",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.groq",
                config_class="llama_stack.providers.remote.inference.groq.GroqConfig",
                provider_data_validator="llama_stack.providers.remote.inference.groq.config.GroqProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="fireworks-openai-compat",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.fireworks_openai_compat",
                config_class="llama_stack.providers.remote.inference.fireworks_openai_compat.config.FireworksCompatConfig",
                provider_data_validator="llama_stack.providers.remote.inference.fireworks_openai_compat.config.FireworksProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="together-openai-compat",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.together_openai_compat",
                config_class="llama_stack.providers.remote.inference.together_openai_compat.config.TogetherCompatConfig",
                provider_data_validator="llama_stack.providers.remote.inference.together_openai_compat.config.TogetherProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="groq-openai-compat",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.groq_openai_compat",
                config_class="llama_stack.providers.remote.inference.groq_openai_compat.config.GroqCompatConfig",
                provider_data_validator="llama_stack.providers.remote.inference.groq_openai_compat.config.GroqProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="sambanova-openai-compat",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.sambanova_openai_compat",
                config_class="llama_stack.providers.remote.inference.sambanova_openai_compat.config.SambaNovaCompatConfig",
                provider_data_validator="llama_stack.providers.remote.inference.sambanova_openai_compat.config.SambaNovaProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="cerebras-openai-compat",
                pip_packages=["litellm"],
                module="llama_stack.providers.remote.inference.cerebras_openai_compat",
                config_class="llama_stack.providers.remote.inference.cerebras_openai_compat.config.CerebrasCompatConfig",
                provider_data_validator="llama_stack.providers.remote.inference.cerebras_openai_compat.config.CerebrasProviderDataValidator",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="sambanova",
                pip_packages=[
                    "openai",
                ],
                module="llama_stack.providers.remote.inference.sambanova",
                config_class="llama_stack.providers.remote.inference.sambanova.SambaNovaImplConfig",
            ),
        ),
        remote_provider_spec(
            api=Api.inference,
            adapter=AdapterSpec(
                adapter_type="passthrough",
                pip_packages=[],
                module="llama_stack.providers.remote.inference.passthrough",
                config_class="llama_stack.providers.remote.inference.passthrough.PassthroughImplConfig",
                provider_data_validator="llama_stack.providers.remote.inference.passthrough.PassthroughProviderDataValidator",
            ),
        ),
    ]
