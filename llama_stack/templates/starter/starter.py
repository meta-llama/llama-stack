# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.apis.models.models import ModelType
from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from llama_stack.providers.remote.inference.anthropic.config import AnthropicConfig
from llama_stack.providers.remote.inference.anthropic.models import (
    MODEL_ENTRIES as ANTHROPIC_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.fireworks.config import FireworksImplConfig
from llama_stack.providers.remote.inference.fireworks.models import (
    MODEL_ENTRIES as FIREWORKS_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.gemini.config import GeminiConfig
from llama_stack.providers.remote.inference.gemini.models import (
    MODEL_ENTRIES as GEMINI_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.remote.inference.groq.models import (
    MODEL_ENTRIES as GROQ_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.ollama.config import OllamaImplConfig
from llama_stack.providers.remote.inference.ollama.models import (
    MODEL_ENTRIES as OLLAMA_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.openai.models import (
    MODEL_ENTRIES as OPENAI_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.sambanova.config import SambaNovaImplConfig
from llama_stack.providers.remote.inference.sambanova.models import (
    MODEL_ENTRIES as SAMBANOVA_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.together.config import TogetherImplConfig
from llama_stack.providers.remote.inference.together.models import (
    MODEL_ENTRIES as TOGETHER_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.vllm import VLLMInferenceAdapterConfig
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
)


def get_inference_providers() -> tuple[list[Provider], dict[str, list[ProviderModelEntry]]]:
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
            "together",
            TOGETHER_MODEL_ENTRIES,
            TogetherImplConfig.sample_run_config(api_key="${env.TOGETHER_API_KEY:}"),
        ),
        (
            "ollama",
            OLLAMA_MODEL_ENTRIES,
            OllamaImplConfig.sample_run_config(),
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
        (
            "groq",
            GROQ_MODEL_ENTRIES,
            GroqConfig.sample_run_config(api_key="${env.GROQ_API_KEY:}"),
        ),
        (
            "sambanova",
            SAMBANOVA_MODEL_ENTRIES,
            SambaNovaImplConfig.sample_run_config(api_key="${env.SAMBANOVA_API_KEY:}"),
        ),
        (
            "vllm",
            [],
            VLLMInferenceAdapterConfig.sample_run_config(
                url="${env.VLLM_URL:http://localhost:8000/v1}",
            ),
        ),
    ]
    inference_providers = []
    available_models = {}
    for provider_id, model_entries, config in providers:
        inference_providers.append(
            Provider(
                provider_id=provider_id,
                provider_type=f"remote::{provider_id}",
                config=config,
            )
        )
        available_models[provider_id] = model_entries
    return inference_providers, available_models


def get_distribution_template() -> DistributionTemplate:
    inference_providers, available_models = get_inference_providers()
    providers = {
        "inference": ([p.provider_type for p in inference_providers] + ["inline::sentence-transformers"]),
        "vector_io": ["inline::sqlite-vec", "remote::chromadb", "remote::pgvector"],
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
    name = "starter"

    vector_io_providers = [
        Provider(
            provider_id="sqlite-vec",
            provider_type="inline::sqlite-vec",
            config=SQLiteVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
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
    files_provider = Provider(
        provider_id="meta-reference-files",
        provider_type="inline::localfs",
        config=LocalfsFilesImplConfig.sample_run_config(f"~/.llama/distributions/{name}"),
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
    ]
    embedding_model = ModelInput(
        model_id="all-MiniLM-L6-v2",
        provider_id=embedding_provider.provider_id,
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
        },
    )

    default_models = get_model_registry(available_models)

    postgres_store = PostgresSqlStoreConfig.sample_run_config()
    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Quick start template for running Llama Stack with several popular providers",
        container_image=None,
        template_path=None,
        providers=providers,
        available_models_by_provider=available_models,
        additional_pip_packages=postgres_store.pip_packages,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": inference_providers + [embedding_provider],
                    "vector_io": vector_io_providers,
                    "files": [files_provider],
                },
                default_models=default_models + [embedding_model],
                default_tool_groups=default_tool_groups,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
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
                "VLLM URL",
            ),
        },
    )
