# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import ModelInput, Provider, ShieldInput
from llama_stack.providers.inline.memory.faiss.config import FaissImplConfig
from llama_stack.providers.remote.inference.vllm import VLLMInferenceAdapterConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::vllm"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
    }
    name = "remote-vllm"
    inference_provider = Provider(
        provider_id="vllm-inference",
        provider_type="remote::vllm",
        config=VLLMInferenceAdapterConfig.sample_run_config(
            url="${env.VLLM_URL}",
        ),
    )
    memory_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissImplConfig.sample_run_config(f"distributions/{name}"),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="vllm-inference",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="vllm-safety",
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use (an external) vLLM server for running LLM inference",
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        default_models=[inference_model, safety_model],
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "memory": [memory_provider],
                },
                default_models=[inference_model],
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [
                        inference_provider,
                        Provider(
                            provider_id="vllm-safety",
                            provider_type="remote::vllm",
                            config=VLLMInferenceAdapterConfig.sample_run_config(
                                url="${env.SAFETY_VLLM_URL}",
                            ),
                        ),
                    ],
                    "memory": [memory_provider],
                },
                default_models=[
                    inference_model,
                    safety_model,
                ],
                default_shields=[ShieldInput(shield_id="${env.SAFETY_MODEL}")],
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "INFERENCE_MODEL": (
                "meta-llama/Llama-3.2-3B-Instruct",
                "Inference model loaded into the vLLM server",
            ),
            "VLLM_URL": (
                "http://host.docker.internal:5100}/v1",
                "URL of the vLLM server with the main inference model",
            ),
            "MAX_TOKENS": (
                "4096",
                "Maximum number of tokens for generation",
            ),
            "SAFETY_VLLM_URL": (
                "http://host.docker.internal:5101/v1",
                "URL of the vLLM server with the safety model",
            ),
            "SAFETY_MODEL": (
                "meta-llama/Llama-Guard-3-1B",
                "Name of the safety (Llama-Guard) model to use",
            ),
        },
    )
