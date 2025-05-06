# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import Provider, ToolGroupInput
from llama_stack.providers.remote.inference.watsonx import WatsonXConfig
from llama_stack.providers.remote.inference.watsonx.models import MODEL_ENTRIES
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings, get_model_registry


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::watsonx"],
        "vector_io": ["inline::faiss"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
        "tool_runtime": [
            "remote::brave-search",
            "remote::tavily-search",
            "inline::rag-runtime",
            "remote::model-context-protocol",
        ],
    }

    inference_provider = Provider(
        provider_id="watsonx",
        provider_type="remote::watsonx",
        config=WatsonXConfig.sample_run_config(),
    )

    available_models = {
        "watsonx": MODEL_ENTRIES,
    }
    default_tool_groups = [
        ToolGroupInput(
            toolgroup_id="builtin::websearch",
            provider_id="tavily-search",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::rag",
            provider_id="rag-runtime",
        ),
    ]

    default_models = get_model_registry(available_models)
    return DistributionTemplate(
        name="watsonx",
        distro_type="remote_hosted",
        description="Use watsonx for running LLM inference",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        available_models_by_provider=available_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                },
                default_models=default_models,
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "WATSONX_API_KEY": (
                "",
                "watsonx API Key",
            ),
            "WATSONX_PROJECT_ID": (
                "",
                "watsonx Project ID",
            ),
        },
    )
