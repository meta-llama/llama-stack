# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from llama_stack.apis.models import ModelType
from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ProviderSpec,
    ToolGroupInput,
)
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.inline.vector_io.milvus.config import (
    MilvusVectorIOConfig,
)
from llama_stack.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from llama_stack.providers.registry.inference import available_providers
from llama_stack.providers.remote.inference.anthropic.models import (
    MODEL_ENTRIES as ANTHROPIC_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.bedrock.models import (
    MODEL_ENTRIES as BEDROCK_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.cerebras.models import (
    MODEL_ENTRIES as CEREBRAS_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.databricks.databricks import (
    MODEL_ENTRIES as DATABRICKS_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.fireworks.models import (
    MODEL_ENTRIES as FIREWORKS_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.gemini.models import (
    MODEL_ENTRIES as GEMINI_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.groq.models import (
    MODEL_ENTRIES as GROQ_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.nvidia.models import (
    MODEL_ENTRIES as NVIDIA_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.openai.models import (
    MODEL_ENTRIES as OPENAI_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.runpod.runpod import (
    MODEL_ENTRIES as RUNPOD_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.sambanova.models import (
    MODEL_ENTRIES as SAMBANOVA_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.together.models import (
    MODEL_ENTRIES as TOGETHER_MODEL_ENTRIES,
)
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from llama_stack.providers.utils.inference.model_registry import ProviderModelEntry
from llama_stack.providers.utils.sqlstore.sqlstore import PostgresSqlStoreConfig
from llama_stack.templates.template import (
    DistributionTemplate,
    RunConfigSettings,
    get_model_registry,
    get_shield_registry,
)


def _get_model_entries_for_provider(provider_type: str) -> list[ProviderModelEntry]:
    """Get model entries for a specific provider type."""
    model_entries_map = {
        "openai": OPENAI_MODEL_ENTRIES,
        "fireworks": FIREWORKS_MODEL_ENTRIES,
        "together": TOGETHER_MODEL_ENTRIES,
        "anthropic": ANTHROPIC_MODEL_ENTRIES,
        "gemini": GEMINI_MODEL_ENTRIES,
        "groq": GROQ_MODEL_ENTRIES,
        "sambanova": SAMBANOVA_MODEL_ENTRIES,
        "cerebras": CEREBRAS_MODEL_ENTRIES,
        "bedrock": BEDROCK_MODEL_ENTRIES,
        "databricks": DATABRICKS_MODEL_ENTRIES,
        "nvidia": NVIDIA_MODEL_ENTRIES,
        "runpod": RUNPOD_MODEL_ENTRIES,
    }

    # Special handling for providers with dynamic model entries
    if provider_type == "ollama":
        return [
            ProviderModelEntry(
                provider_model_id="${env.OLLAMA_INFERENCE_MODEL:=__disabled__}",
                model_type=ModelType.llm,
            ),
            ProviderModelEntry(
                provider_model_id="${env.SAFETY_MODEL:=__disabled__}",
                model_type=ModelType.llm,
            ),
            ProviderModelEntry(
                provider_model_id="${env.OLLAMA_EMBEDDING_MODEL:=__disabled__}",
                model_type=ModelType.embedding,
                metadata={
                    "embedding_dimension": "${env.OLLAMA_EMBEDDING_DIMENSION:=384}",
                },
            ),
        ]
    elif provider_type == "vllm":
        return [
            ProviderModelEntry(
                provider_model_id="${env.VLLM_INFERENCE_MODEL:=__disabled__}",
                model_type=ModelType.llm,
            ),
        ]

    return model_entries_map.get(provider_type, [])


def _get_model_safety_entries_for_provider(provider_type: str) -> list[ProviderModelEntry]:
    """Get model entries for a specific provider type."""
    safety_model_entries_map = {
        "ollama": [
            ProviderModelEntry(
                provider_model_id="${env.SAFETY_MODEL:=__disabled__}",
                model_type=ModelType.llm,
            ),
        ],
    }

    return safety_model_entries_map.get(provider_type, [])


def _get_config_for_provider(provider_spec: ProviderSpec) -> dict[str, Any]:
    """Get configuration for a provider using its adapter's config class."""
    config_class = instantiate_class_type(provider_spec.config_class)

    if hasattr(config_class, "sample_run_config"):
        config: dict[str, Any] = config_class.sample_run_config()
        return config
    return {}


def get_remote_inference_providers() -> tuple[list[Provider], dict[str, list[ProviderModelEntry]]]:
    all_providers = available_providers()

    # Filter out inline providers and watsonx - the starter distro only exposes remote providers
    remote_providers = [
        provider
        for provider in all_providers
        # TODO: re-add once the Python 3.13 issue is fixed
        # discussion: https://github.com/meta-llama/llama-stack/pull/2327#discussion_r2156883828
        if hasattr(provider, "adapter") and provider.adapter.adapter_type != "watsonx"
    ]

    providers = []
    available_models = {}

    for provider_spec in remote_providers:
        provider_type = provider_spec.adapter.adapter_type

        # Build the environment variable name for enabling this provider
        env_var = f"ENABLE_{provider_type.upper().replace('-', '_').replace('::', '_')}"
        model_entries = _get_model_entries_for_provider(provider_type)
        config = _get_config_for_provider(provider_spec)
        providers.append(
            (
                f"${{env.{env_var}:=__disabled__}}",
                provider_type,
                model_entries,
                config,
            )
        )
        available_models[f"${{env.{env_var}:=__disabled__}}"] = model_entries

    inference_providers = []
    for provider_id, provider_type, model_entries, config in providers:
        inference_providers.append(
            Provider(
                provider_id=provider_id,
                provider_type=f"remote::{provider_type}",
                config=config,
            )
        )
        available_models[provider_id] = model_entries
    return inference_providers, available_models


# build a list of shields for all possible providers
def get_safety_models_for_providers(providers: list[Provider]) -> dict[str, list[ProviderModelEntry]]:
    available_models = {}
    for provider in providers:
        provider_type = provider.provider_type.split("::")[1]
        safety_model_entries = _get_model_safety_entries_for_provider(provider_type)
        if len(safety_model_entries) == 0:
            continue

        env_var = f"ENABLE_{provider_type.upper().replace('-', '_').replace('::', '_')}"
        provider_id = f"${{env.{env_var}:=__disabled__}}"

        available_models[provider_id] = safety_model_entries

    return available_models


def get_distribution_template() -> DistributionTemplate:
    remote_inference_providers, available_models = get_remote_inference_providers()

    name = "starter"

    vector_io_providers = [
        Provider(
            provider_id="${env.ENABLE_FAISS:=faiss}",
            provider_type="inline::faiss",
            config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
        ),
        Provider(
            provider_id="${env.ENABLE_SQLITE_VEC:=__disabled__}",
            provider_type="inline::sqlite-vec",
            config=SQLiteVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
        ),
        Provider(
            provider_id="${env.ENABLE_MILVUS:=__disabled__}",
            provider_type="inline::milvus",
            config=MilvusVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
        ),
        Provider(
            provider_id="${env.ENABLE_CHROMADB:=__disabled__}",
            provider_type="remote::chromadb",
            config=ChromaVectorIOConfig.sample_run_config(url="${env.CHROMADB_URL:=}"),
        ),
        Provider(
            provider_id="${env.ENABLE_PGVECTOR:=__disabled__}",
            provider_type="remote::pgvector",
            config=PGVectorVectorIOConfig.sample_run_config(
                f"~/.llama/distributions/{name}",
                db="${env.PGVECTOR_DB:=}",
                user="${env.PGVECTOR_USER:=}",
                password="${env.PGVECTOR_PASSWORD:=}",
            ),
        ),
    ]

    providers = {
        "inference": ([p.provider_type for p in remote_inference_providers] + ["inline::sentence-transformers"]),
        "vector_io": ([p.provider_type for p in vector_io_providers]),
        "files": ["inline::localfs"],
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
    files_provider = Provider(
        provider_id="meta-reference-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    embedding_provider = Provider(
        provider_id="${env.ENABLE_SENTENCE_TRANSFORMERS:=sentence-transformers}",
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
    ]
    embedding_model = ModelInput(
        model_id="all-MiniLM-L6-v2",
        provider_id=embedding_provider.provider_id,
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
        },
    )

    default_models, ids_conflict_in_models = get_model_registry(available_models)

    available_safety_models = get_safety_models_for_providers(remote_inference_providers)
    shields = get_shield_registry(available_safety_models, ids_conflict_in_models)

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Quick start template for running Llama Stack with several popular providers",
        container_image=None,
        template_path=None,
        providers=providers,
        available_models_by_provider=available_models,
        additional_pip_packages=PostgresSqlStoreConfig.pip_packages(),
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": remote_inference_providers + [embedding_provider],
                    "vector_io": vector_io_providers,
                    "files": [files_provider],
                },
                default_models=default_models + [embedding_model],
                default_tool_groups=default_tool_groups,
                # TODO: add a way to enable/disable shields on the fly
                default_shields=shields,
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "8321",
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
            "GROQ_API_KEY": (
                "",
                "Groq API Key",
            ),
            "ANTHROPIC_API_KEY": (
                "",
                "Anthropic API Key",
            ),
            "GEMINI_API_KEY": (
                "",
                "Gemini API Key",
            ),
            "SAMBANOVA_API_KEY": (
                "",
                "SambaNova API Key",
            ),
            "VLLM_URL": (
                "http://localhost:8000/v1",
                "vLLM URL",
            ),
            "VLLM_INFERENCE_MODEL": (
                "",
                "Optional vLLM Inference Model to register on startup",
            ),
            "OLLAMA_URL": (
                "http://localhost:11434",
                "Ollama URL",
            ),
            "OLLAMA_INFERENCE_MODEL": (
                "",
                "Optional Ollama Inference Model to register on startup",
            ),
            "OLLAMA_EMBEDDING_MODEL": (
                "",
                "Optional Ollama Embedding Model to register on startup",
            ),
            "OLLAMA_EMBEDDING_DIMENSION": (
                "384",
                "Ollama Embedding Dimension",
            ),
        },
    )
