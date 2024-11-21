# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.distribution.datatypes import ModelInput, Provider, ShieldInput
from llama_stack.providers.remote.inference.tgi import InferenceEndpointImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::hf::endpoint"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
    }

    inference_provider = Provider(
        provider_id="hf-endpoint",
        provider_type="remote::hf::endpoint",
        config=InferenceEndpointImplConfig.sample_run_config(),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="hf-endpoint",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="hf-endpoint-safety",
    )

    return DistributionTemplate(
        name="hf-endpoint",
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
                },
                default_models=[inference_model],
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [
                        inference_provider,
                        Provider(
                            provider_id="hf-endpoint-safety",
                            provider_type="remote::hf::endpoint",
                            config=InferenceEndpointImplConfig.sample_run_config(
                                endpoint_name="${env.SAFETY_INFERENCE_ENDPOINT_NAME}",
                            ),
                        ),
                    ]
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
            "INFERENCE_ENDPOINT_NAME": (
                "",
                "HF Inference endpoint name for the main inference model",
            ),
            "SAFETY_INFERENCE_ENDPOINT_NAME": (
                "",
                "HF Inference endpoint for the safety model",
            ),
            "INFERENCE_MODEL": (
                "meta-llama/Llama-3.2-3B-Instruct",
                "Inference model served by the HF Inference Endpoint",
            ),
            "SAFETY_MODEL": (
                "meta-llama/Llama-Guard-3-1B",
                "Safety model served by the HF Inference Endpoint",
            ),
        },
    )
