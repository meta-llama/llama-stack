# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import ModelInput, Provider, ShieldInput, ToolGroupInput
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.remote.inference.nvidia import NVIDIAConfig
from llama_stack.providers.remote.inference.nvidia.models import _MODEL_ENTRIES
from llama_stack.providers.remote.safety.nvidia import NVIDIASafetyConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::nvidia"],
        "vector_io": ["inline::faiss"],
        "safety": ["remote::nvidia"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["inline::localfs"],
        "scoring": ["inline::basic"],
        "tool_runtime": ["inline::rag-runtime"],
    }

    inference_provider = Provider(
        provider_id="nvidia",
        provider_type="remote::nvidia",
        config=NVIDIAConfig.sample_run_config(),
    )
    safety_provider = Provider(
        provider_id="nvidia",
        provider_type="remote::nvidia",
        config=NVIDIASafetyConfig.sample_run_config(),
    )
    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="nvidia",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="nvidia",
    )

    core_model_to_hf_repo = {m.descriptor(): m.huggingface_repo for m in all_registered_models()}
    default_models = [
        ModelInput(
            model_id=core_model_to_hf_repo[m.llama_model] if m.llama_model else m.provider_model_id,
            provider_model_id=m.provider_model_id,
            provider_id="nvidia",
            model_type=m.model_type,
            metadata=m.metadata,
        )
        for m in _MODEL_ENTRIES
    ]
    default_tool_groups = [
        ToolGroupInput(
            toolgroup_id="builtin::rag",
            provider_id="rag-runtime",
        ),
    ]

    return DistributionTemplate(
        name="nvidia",
        distro_type="remote_hosted",
        description="Use NVIDIA NIM for running LLM inference and safety",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        default_models=default_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                },
                default_models=default_models,
                default_tool_groups=default_tool_groups,
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [
                        inference_provider,
                        safety_provider,
                    ]
                },
                default_models=[inference_model, safety_model],
                default_shields=[ShieldInput(shield_id="${env.SAFETY_MODEL}", provider_id="nvidia")],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "NVIDIA_API_KEY": (
                "",
                "NVIDIA API Key",
            ),
            "GUARDRAILS_SERVICE_URL": (
                "http://0.0.0.0:7331",
                "URL for the NeMo Guardrails Service",
            ),
            "INFERENCE_MODEL": (
                "Llama3.1-8B-Instruct",
                "Inference model",
            ),
            "SAFETY_MODEL": (
                "meta/llama-3.1-8b-instruct",
                "Name of the model to use for safety",
            ),
        },
    )
