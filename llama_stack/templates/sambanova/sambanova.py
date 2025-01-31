# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_models.sku_list import all_registered_models

from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.providers.remote.inference.sambanova import SambaNovaImplConfig
from llama_stack.providers.remote.inference.sambanova.sambanova import MODEL_ALIASES

from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::sambanova"],
        "vector_io": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "tool_runtime": [
            "remote::brave-search",
            "remote::tavily-search",
            "inline::code-interpreter",
            "inline::rag-runtime",
        ],
    }
    name = "sambanova"

    inference_provider = Provider(
        provider_id=name,
        provider_type=f"remote::{name}",
        config=SambaNovaImplConfig.sample_run_config(),
    )

    core_model_to_hf_repo = {
        m.descriptor(): m.huggingface_repo for m in all_registered_models()
    }
    default_models = [
        ModelInput(
            model_id=core_model_to_hf_repo[m.llama_model],
            provider_model_id=m.provider_model_id,
            provider_id=name,
        )
        for m in MODEL_ALIASES
    ]

    default_tool_groups = [
        ToolGroupInput(
            toolgroup_id="builtin::websearch",
            provider_id="tavily-search",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::rag",
            provider_id="rag-runtime",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::code_interpreter",
            provider_id="code-interpreter",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use SambaNova.AI for running LLM inference",
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
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "SAMBANOVA_API_KEY": (
                "",
                "SambaNova.AI API Key",
            ),
        },
    )
