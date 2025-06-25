# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.apis.models import ModelType
from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ToolGroupInput,
)
from llama_stack.providers.inline.files.localfs.config import LocalfsFilesImplConfig
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.post_training.huggingface import HuggingFacePostTrainingConfig
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.inline.vector_io.sqlite_vec.config import (
    SQLiteVectorIOConfig,
)
from llama_stack.providers.remote.inference.anthropic.config import AnthropicConfig
from llama_stack.providers.remote.inference.anthropic.models import (
    MODEL_ENTRIES as ANTHROPIC_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.cerebras.config import CerebrasImplConfig
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
from llama_stack.providers.remote.inference.llama_openai_compat.config import (
    LlamaCompatConfig,
)
from llama_stack.providers.remote.inference.nvidia.config import NVIDIAConfig
from llama_stack.providers.remote.inference.ollama.config import OllamaImplConfig
from llama_stack.providers.remote.inference.openai.config import OpenAIConfig
from llama_stack.providers.remote.inference.openai.models import (
    MODEL_ENTRIES as OPENAI_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.passthrough.config import (
    PassthroughImplConfig,
)
from llama_stack.providers.remote.inference.sambanova.config import SambaNovaImplConfig
from llama_stack.providers.remote.inference.sambanova.models import (
    MODEL_ENTRIES as SAMBANOVA_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.tgi import InferenceEndpointImplConfig
from llama_stack.providers.remote.inference.tgi.config import InferenceAPIImplConfig, TGIImplConfig
from llama_stack.providers.remote.inference.together.config import TogetherImplConfig
from llama_stack.providers.remote.inference.together.models import (
    MODEL_ENTRIES as TOGETHER_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.vllm import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from llama_stack.providers.utils.bedrock.config import BedrockBaseConfig
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
            "${env.ENABLE_OPENAI:=__disabled__}",
            "openai",
            OPENAI_MODEL_ENTRIES,
            OpenAIConfig.sample_run_config(api_key="${env.OPENAI_API_KEY:+}"),
        ),
        (
            "${env.ENABLE_FIREWORKS:=__disabled__}",
            "fireworks",
            FIREWORKS_MODEL_ENTRIES,
            FireworksImplConfig.sample_run_config(api_key="${env.FIREWORKS_API_KEY:+}"),
        ),
        (
            "${env.ENABLE_TOGETHER:=__disabled__}",
            "together",
            TOGETHER_MODEL_ENTRIES,
            TogetherImplConfig.sample_run_config(api_key="${env.TOGETHER_API_KEY:+}"),
        ),
        (
            "${env.ENABLE_OLLAMA:=__disabled__}",
            "ollama",
            [
                ProviderModelEntry(
                    provider_model_id="${env.OLLAMA_INFERENCE_MODEL:=__disabled__}",
                    model_type=ModelType.llm,
                ),
                ProviderModelEntry(
                    provider_model_id="${env.OLLAMA_EMBEDDING_MODEL:=__disabled__}",
                    model_type=ModelType.embedding,
                    metadata={
                        "embedding_dimension": "${env.OLLAMA_EMBEDDING_DIMENSION:=384}",
                    },
                ),
            ],
            OllamaImplConfig.sample_run_config(
                url="${env.OLLAMA_URL:=http://localhost:11434}", raise_on_connect_error=False
            ),
        ),
        (
            "${env.ENABLE_ANTHROPIC:=__disabled__}",
            "anthropic",
            ANTHROPIC_MODEL_ENTRIES,
            AnthropicConfig.sample_run_config(api_key="${env.ANTHROPIC_API_KEY:+}"),
        ),
        (
            "${env.ENABLE_GEMINI:=__disabled__}",
            "gemini",
            GEMINI_MODEL_ENTRIES,
            GeminiConfig.sample_run_config(api_key="${env.GEMINI_API_KEY:+}"),
        ),
        (
            "${env.ENABLE_GROQ:=__disabled__}",
            "groq",
            GROQ_MODEL_ENTRIES,
            GroqConfig.sample_run_config(api_key="${env.GROQ_API_KEY:+}"),
        ),
        (
            "${env.ENABLE_SAMBANOVA:=__disabled__}",
            "sambanova",
            SAMBANOVA_MODEL_ENTRIES,
            SambaNovaImplConfig.sample_run_config(api_key="${env.SAMBANOVA_API_KEY:+}"),
        ),
        (
            "${env.ENABLE_VLLM:=__disabled__}",
            "vllm",
            [
                ProviderModelEntry(
                    provider_model_id="${env.VLLM_INFERENCE_MODEL:=__disabled__}",
                    model_type=ModelType.llm,
                ),
            ],
            VLLMInferenceAdapterConfig.sample_run_config(
                url="${env.VLLM_URL:=http://localhost:8000/v1}",
            ),
        ),
        (
            "${env.ENABLE_TGI:=__disabled__}",
            "tgi",
            [],
            TGIImplConfig.sample_run_config(
                url="${env.TGI_URL:+}",
                endpoint_name="${env.INFERENCE_ENDPOINT_NAME:+}",
            ),
        ),
        # TODO: re-add once the Python 3.13 issue is fixed
        # discussion: https://github.com/meta-llama/llama-stack/pull/2327#discussion_r2156883828
        # (
        #     "watsonx",
        #     [],
        #     WatsonXConfig.sample_run_config(api_key="${env.WATSONX_API_KEY:}"),
        # ),
        (
            "${env.ENABLE_CEREBRAS:=__disabled__}",
            "cerebras",
            [],
            CerebrasImplConfig.sample_run_config(api_key="${env.CEREBRAS_API_KEY:+}"),
        ),
        (
            "${env.ENABLE_LLAMA_OPENAI_COMPAT:=__disabled__}",
            "llama-openai-compat",
            [],
            LlamaCompatConfig.sample_run_config(api_key="${env.LLAMA_API_KEY:+:}"),
        ),
        (
            "${env.ENABLE_NVIDIA:=__disabled__}",
            "nvidia",
            [],
            NVIDIAConfig.sample_run_config(
                api_key="${env.NVIDIA_API_KEY:+}",
                url="${env.NVIDIA_BASE_URL:__disabled__}",
            ),
        ),
        (
            "${env.ENABLE_HF_SERVERLESS:=__disabled__}",
            "hf::serverless",
            [],
            InferenceAPIImplConfig.sample_run_config(
                api_token="${env.HF_API_TOKEN:+:}",
                repo="${env.INFERENCE_MODEL:+:}",
            ),
        ),
        (
            "${env.ENABLE_HF_ENDPOINT:=__disabled__}",
            "hf::endpoint",
            [],
            InferenceEndpointImplConfig.sample_run_config(
                api_token="${env.HF_API_TOKEN:+:}",
                endpoint_name="${env.INFERENCE_ENDPOINT_NAME:+:}",
            ),
        ),
        (
            "${env.ENABLE_BEDROCK:=__disabled__}",
            "bedrock",
            [],
            BedrockBaseConfig.sample_run_config(
                aws_access_key_id="${env.AWS_ACCESS_KEY_ID:+}",
                aws_secret_access_key="${env.AWS_SECRET_ACCESS_KEY:+}",
                aws_session_token="${env.AWS_SESSION_TOKEN:+}",
                region_name="${env.AWS_DEFAULT_REGION:+}",
            ),
        ),
        (
            "${env.ENABLE_PASSTHROUGH:=__disabled__}",
            "passthrough",
            [],
            PassthroughImplConfig.sample_run_config(
                url="${env.PASSTHROUGH_URL:+:}", api_key="${env.PASSTHROUGH_API_KEY:+:}"
            ),
        ),
    ]
    inference_providers = []
    available_models = {}
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


def get_distribution_template() -> DistributionTemplate:
    inference_providers, available_models = get_inference_providers()
    providers = {
        "inference": ([p.provider_type for p in inference_providers] + ["inline::sentence-transformers"]),
        "vector_io": ["inline::sqlite-vec", "remote::chromadb", "remote::pgvector"],
        "files": ["inline::localfs"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "post_training": ["inline::huggingface"],
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
            provider_id="${env.ENABLE_CHROMADB:=__disabled__}",
            provider_type="remote::chromadb",
            config=ChromaVectorIOConfig.sample_run_config(url="${env.CHROMADB_URL:+}"),
        ),
        Provider(
            provider_id="${env.ENABLE_PGVECTOR:=__disabled__}",
            provider_type="remote::pgvector",
            config=PGVectorVectorIOConfig.sample_run_config(
                db="${env.PGVECTOR_DB:+}",
                user="${env.PGVECTOR_USER:+}",
                password="${env.PGVECTOR_PASSWORD:+}",
            ),
        ),
    ]
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
    post_training_provider = Provider(
        provider_id="huggingface",
        provider_type="inline::huggingface",
        config=HuggingFacePostTrainingConfig.sample_run_config(f"~/.llama/distributions/{name}"),
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
                    "post_training": [post_training_provider],
                },
                default_models=default_models + [embedding_model],
                default_tool_groups=default_tool_groups,
                # TODO: add a way to enable/disable shields on the fly
                # default_shields=[
                #     ShieldInput(provider_id="llama-guard", shield_id="${env.SAFETY_MODEL:=meta-llama/Llama-Guard-3-8B}")
                # ],
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
