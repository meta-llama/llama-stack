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
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from llama_stack.providers.remote.inference.cerebras.models import MODEL_ENTRIES as CEREBRAS_MODEL_ENTRIES
from llama_stack.providers.remote.inference.cerebras_openai_compat.config import CerebrasCompatConfig
from llama_stack.providers.remote.inference.fireworks.models import (
    MODEL_ENTRIES as FIREWORKS_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.fireworks_openai_compat.config import FireworksCompatConfig
from llama_stack.providers.remote.inference.groq.models import (
    MODEL_ENTRIES as GROQ_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.groq_openai_compat.config import GroqCompatConfig
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.openai.models import (
    MODEL_ENTRIES as OPENAI_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.sambanova.models import MODEL_ENTRIES as SAMBANOVA_MODEL_ENTRIES
from llama_stack.providers.remote.inference.sambanova_openai_compat.config import SambaNovaCompatConfig
from llama_stack.providers.remote.inference.together.models import (
    MODEL_ENTRIES as TOGETHER_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.together_openai_compat.config import TogetherCompatConfig
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from llama_stack.providers.utils.inference.model_registry import ProviderModelEntry
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
            "fireworks-openai-compat",
            FIREWORKS_MODEL_ENTRIES,
            FireworksCompatConfig.sample_run_config(api_key="${env.FIREWORKS_API_KEY:}"),
        ),
        (
            "together-openai-compat",
            TOGETHER_MODEL_ENTRIES,
            TogetherCompatConfig.sample_run_config(api_key="${env.TOGETHER_API_KEY:}"),
        ),
        (
            "groq-openai-compat",
            GROQ_MODEL_ENTRIES,
            GroqCompatConfig.sample_run_config(api_key="${env.GROQ_API_KEY:}"),
        ),
        (
            "sambanova-openai-compat",
            SAMBANOVA_MODEL_ENTRIES,
            SambaNovaCompatConfig.sample_run_config(api_key="${env.SAMBANOVA_API_KEY:}"),
        ),
        (
            "cerebras-openai-compat",
            CEREBRAS_MODEL_ENTRIES,
            CerebrasCompatConfig.sample_run_config(api_key="${env.CEREBRAS_API_KEY:}"),
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
    name = "verification"

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
    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Distribution for running e2e tests in CI",
        container_image=None,
        template_path=None,
        providers=providers,
        available_models_by_provider=available_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": inference_providers + [embedding_provider],
                    "vector_io": vector_io_providers,
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
        },
    )
