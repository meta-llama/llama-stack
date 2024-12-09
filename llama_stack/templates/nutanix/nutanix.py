# Copyright (c) Nutanix, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_models.sku_list import all_registered_models

from llama_stack.distribution.datatypes import ModelInput, Provider, ShieldInput
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings

from llama_stack.providers.remote.inference.nutanix import NutanixImplConfig
from llama_stack.providers.remote.inference.nutanix.nutanix import MODEL_ALIASES

def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::nutanix"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
    }
    
    inference_provider = Provider(
        provider_id="nutanix",
        provider_type="remote::nutanix",
        config=NutanixImplConfig.sample_run_config(),
    )
    
    core_model_to_hf_repo = {
        m.descriptor(): m.huggingface_repo for m in all_registered_models()
    }
    default_models = [
        ModelInput(
            model_id=core_model_to_hf_repo[m.llama_model],
            provider_model_id=m.provider_model_id,
        )
        for m in MODEL_ALIASES
    ]

    return DistributionTemplate(
        name="nutanix",
        distro_type="remote::nutanix",
        description="Use Nutanix AI Endpoint for running LLM inference",
        docker_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        default_models=default_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                },
                default_models=default_models,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "NUTANIX_API_KEY": (
                "",
                "Nutanix AI Endpoint API Key",
            ),
        },
    )
