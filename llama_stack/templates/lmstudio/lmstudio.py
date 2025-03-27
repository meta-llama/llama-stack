# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import Provider, ToolGroupInput
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.remote.inference.lmstudio import LMStudioImplConfig
from llama_stack.providers.remote.inference.lmstudio.models import MODEL_ENTRIES
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings, get_model_registry


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::lmstudio"],
        "safety": ["inline::llama-guard"],
        "vector_io": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "agents": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
        "telemetry": ["inline::meta-reference"],
        "tool_runtime": [
            "remote::tavily-search",
            "inline::code-interpreter",
            "inline::rag-runtime",
        ],
    }

    name = "lmstudio"
    lmstudio_provider = Provider(
        provider_id="lmstudio",
        provider_type="remote::lmstudio",
        config=LMStudioImplConfig.sample_run_config(),
    )

    available_models = {
        "lmstudio": MODEL_ENTRIES,
    }
    default_models = get_model_registry(available_models)
    vector_io_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
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
        name="lmstudio",
        distro_type="self_hosted",
        description="Use LM Studio for running LLM inference",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        available_models_by_provider=available_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [lmstudio_provider],
                    "vector_io": [vector_io_provider],
                },
                default_models=default_models,
                default_shields=[],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
        },
    )
