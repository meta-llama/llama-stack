# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.apis.models.models import ModelType
from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.remote.inference.llamacpp.config import LlamaCppImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::llamacpp", "inline::sentence-transformers"],
        "vector_io": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
        "files": ["inline::localfs"],
        "tool_runtime": [
            "remote::brave-search",
            "remote::tavily-search",
            "remote::wolfram-alpha",
            "inline::rag-runtime",
            "remote::model-context-protocol",
        ],
    }
    name = "llamacpp"
    inference_provider = Provider(
        provider_id="llamacpp",
        provider_type="remote::llamacpp",
        config=LlamaCppImplConfig.sample_run_config(),
    )
    sentence_transformers_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config={},
    )
    vector_io_provider_faiss = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    files_provider = Provider(
        provider_id="meta-reference-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="llamacpp",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="llamacpp",
    )
    embedding_model = ModelInput(
        model_id="all-MiniLM-L6-v2",
        provider_id="sentence-transformers",
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
        },
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
            toolgroup_id="builtin::wolfram_alpha",
            provider_id="wolfram-alpha",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use llama.cpp server for running LLM inference",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider, sentence_transformers_provider],
                    "vector_io": [vector_io_provider_faiss],
                    "files": [files_provider],
                },
                default_models=[inference_model, embedding_model],
                default_tool_groups=default_tool_groups,
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider, sentence_transformers_provider],
                    "vector_io": [vector_io_provider_faiss],
                    "files": [files_provider],
                    "safety": [
                        Provider(
                            provider_id="llama-guard",
                            provider_type="inline::llama-guard",
                            config={},
                        ),
                    ],
                },
                default_models=[
                    inference_model,
                    safety_model,
                    embedding_model,
                ],
                default_shields=[
                    ShieldInput(
                        shield_id="${env.SAFETY_MODEL}",
                        provider_id="llama-guard",
                    ),
                ],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "8321",
                "Port for the Llama Stack distribution server",
            ),
            "LLAMACPP_URL": (
                "http://localhost:8080",
                "URL of the llama.cpp server (without /v1 suffix)",
            ),
            "LLAMACPP_API_KEY": (
                "",
                "API key for llama.cpp server (leave empty for local servers)",
            ),
            "INFERENCE_MODEL": (
                "llama-model",
                "Inference model identifier for llama.cpp server",
            ),
            "SAFETY_MODEL": (
                "llama-guard",
                "Safety model identifier for llama.cpp server",
            ),
        },
    )
