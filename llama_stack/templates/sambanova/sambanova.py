# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import Provider, ShieldInput, ToolGroupInput
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.remote.inference.sambanova import SambaNovaImplConfig
from llama_stack.providers.remote.inference.sambanova.models import MODEL_ENTRIES
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from llama_stack.templates.template import (
    DistributionTemplate,
    RunConfigSettings,
    get_model_registry,
)


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

    vector_io_providers = [
        Provider(
            provider_id="faiss",
            provider_type="inline::faiss",
            config=FaissVectorIOConfig.sample_run_config(
                __distro_dir__=f"~/.llama/distributions/{name}",
            ),
        ),
        Provider(
            provider_id="${env.ENABLE_CHROMADB+chromadb}",
            provider_type="remote::chromadb",
            config=ChromaVectorIOConfig.sample_run_config(url="${env.CHROMADB_URL:}"),
        ),
        Provider(
            provider_id="${env.ENABLE_PGVECTOR+pgvector}",
            provider_type="remote::pgvector",
            config=PGVectorVectorIOConfig.sample_run_config(
                db="${env.PGVECTOR_DB:}",
                user="${env.PGVECTOR_USER:}",
                password="${env.PGVECTOR_PASSWORD:}",
            ),
        ),
    ]

    available_models = {
        name: MODEL_ENTRIES,
    }
    default_models = get_model_registry(available_models)
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
        available_models_by_provider=available_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "vector_io": vector_io_providers,
                },
                default_models=default_models,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "8321",
                "Port for the Llama Stack distribution server",
            ),
            "SAMBANOVA_API_KEY": (
                "",
                "SambaNova.AI API Key",
            ),
        },
    )
