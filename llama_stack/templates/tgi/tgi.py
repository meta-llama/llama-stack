# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import ModelInput, Provider, ShieldInput
from llama_stack.providers.remote.inference.tgi import TGIImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::tgi"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
    }

    inference_provider = Provider(
        provider_id="tgi-inference",
        provider_type="remote::tgi",
        config=TGIImplConfig.sample_run_config(
            url="${env.TGI_URL}",
        ),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="tgi-inference",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="tgi-safety",
    )

    return DistributionTemplate(
        name="tgi",
        distro_type="self_hosted",
        description="Use (an external) TGI server for running LLM inference",
        docker_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
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
                            provider_id="tgi-safety",
                            provider_type="remote::tgi",
                            config=TGIImplConfig.sample_run_config(
                                url="${env.TGI_SAFETY_URL}",
                            ),
                        ),
                    ],
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
                "Inference model loaded into the TGI server",
            ),
            "TGI_URL": (
                "http://127.0.0.1:8080}/v1",
                "URL of the TGI server with the main inference model",
            ),
            "TGI_SAFETY_URL": (
                "http://127.0.0.1:8081/v1",
                "URL of the TGI server with the safety model",
            ),
            "SAFETY_MODEL": (
                "meta-llama/Llama-Guard-3-1B",
                "Name of the safety (Llama-Guard) model to use",
            ),
        },
    )
