# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.distribution.datatypes import ModelInput, Provider, ShieldInput
from llama_stack.providers.inline.memory.faiss.config import FaissImplConfig
from llama_stack.providers.remote.inference.tgi import InferenceAPIImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::hf::serverless"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
    }

    name = "hf-serverless"
    inference_provider = Provider(
        provider_id="hf-serverless",
        provider_type="remote::hf::serverless",
        config=InferenceAPIImplConfig.sample_run_config(),
    )
    memory_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissImplConfig.sample_run_config(f"distributions/{name}"),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="hf-serverless",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="hf-serverless-safety",
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use (an external) Hugging Face Inference Endpoint for running LLM inference",
        docker_image=None,
        template_path=None,
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
                            provider_id="hf-serverless-safety",
                            provider_type="remote::hf::serverless",
                            config=InferenceAPIImplConfig.sample_run_config(
                                repo="${env.SAFETY_MODEL}",
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
            "HF_API_TOKEN": (
                "hf_...",
                "Hugging Face API token",
            ),
            "INFERENCE_MODEL": (
                "meta-llama/Llama-3.2-3B-Instruct",
                "Inference model to be served by the HF Serverless endpoint",
            ),
            "SAFETY_MODEL": (
                "meta-llama/Llama-Guard-3-1B",
                "Safety model to be served by the HF Serverless endpoint",
            ),
        },
    )
