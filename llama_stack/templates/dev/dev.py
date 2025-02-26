# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Tuple

from llama_stack.apis.models.models import ModelType
from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.vector_io.sqlite_vec.config import SQLiteVectorIOConfig
from llama_stack.providers.remote.inference.anthropic.config import AnthropicConfig
from llama_stack.providers.remote.inference.anthropic.models import MODEL_ENTRIES as ANTHROPIC_MODEL_ENTRIES
from llama_stack.providers.remote.inference.fireworks.config import FireworksImplConfig
from llama_stack.providers.remote.inference.fireworks.models import MODEL_ENTRIES as FIREWORKS_MODEL_ENTRIES
from llama_stack.providers.remote.inference.gemini.config import GeminiConfig
from llama_stack.providers.remote.inference.gemini.models import MODEL_ENTRIES as GEMINI_MODEL_ENTRIES
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.openai.models import MODEL_ENTRIES as OPENAI_MODEL_ENTRIES
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_inference_providers() -> Tuple[List[Provider], List[ModelInput]]:
    # in this template, we allow each API key to be optional
    providers = [
        (
            "openai",
            OPENAI_MODEL_ENTRIES,
            OpenAIConfig.sample_run_config(api_key="${env.OPENAI_API_KEY:}"),
        ),
        (
            "fireworks",
            FIREWORKS_MODEL_ENTRIES,
            FireworksImplConfig.sample_run_config(api_key="${env.FIREWORKS_API_KEY:}"),
        ),
        (
            "anthropic",
            ANTHROPIC_MODEL_ENTRIES,
            AnthropicConfig.sample_run_config(api_key="${env.ANTHROPIC_API_KEY:}"),
        ),
        (
            "gemini",
            GEMINI_MODEL_ENTRIES,
            GeminiConfig.sample_run_config(api_key="${env.GEMINI_API_KEY:}"),
        ),
    ]
    inference_providers = []
    default_models = []
    core_model_to_hf_repo = {m.descriptor(): m.huggingface_repo for m in all_registered_models()}
    for provider_id, model_entries, config in providers:
        inference_providers.append(
            Provider(
                provider_id=provider_id,
                provider_type=f"remote::{provider_id}",
                config=config,
            )
        )
        default_models.extend(
            ModelInput(
                model_id=core_model_to_hf_repo[m.llama_model] if m.llama_model else m.provider_model_id,
                provider_model_id=m.provider_model_id,
                provider_id=provider_id,
                model_type=m.model_type,
                metadata=m.metadata,
            )
            for m in model_entries
        )
    return inference_providers, default_models


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": [
            "remote::openai",
            "remote::fireworks",
            "remote::anthropic",
            "remote::gemini",
            "inline::sentence-transformers",
        ],
        "vector_io": ["inline::sqlite-vec", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
        "tool_runtime": [
            "remote::brave-search",
            "remote::tavily-search",
            "inline::code-interpreter",
            "inline::rag-runtime",
            "remote::model-context-protocol",
        ],
    }
    name = "dev"

    vector_io_provider = Provider(
        provider_id="sqlite-vec",
        provider_type="inline::sqlite-vec",
        config=SQLiteVectorIOConfig.sample_run_config(f"distributions/{name}"),
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
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
    embedding_model = ModelInput(
        model_id="all-MiniLM-L6-v2",
        provider_id=embedding_provider.provider_id,
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
        },
    )
    inference_providers, default_models = get_inference_providers()

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Distribution for running e2e tests in CI",
        container_image=None,
        template_path=None,
        providers=providers,
        default_models=[],
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": inference_providers + [embedding_provider],
                    "vector_io": [vector_io_provider],
                },
                default_models=default_models + [embedding_model],
                default_tool_groups=default_tool_groups,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "FIREWORKS_API_KEY": (
                "",
                "Fireworks API Key",
            ),
            "OPENAI_API_KEY": (
                "",
                "OpenAI API Key",
            ),
        },
    )
