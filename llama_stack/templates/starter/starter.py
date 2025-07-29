# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any

from llama_stack.distribution.datatypes import (
    BuildProvider,
    Provider,
    ProviderSpec,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.providers.datatypes import RemoteProviderSpec
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
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.config import (
    PGVectorVectorIOConfig,
)
from llama_stack.providers.utils.sqlstore.sqlstore import PostgresSqlStoreConfig
from llama_stack.templates.template import (
    DistributionTemplate,
    RunConfigSettings,
)


def _get_config_for_provider(provider_spec: ProviderSpec) -> dict[str, Any]:
    """Get configuration for a provider using its adapter's config class."""
    config_class = instantiate_class_type(provider_spec.config_class)

    if hasattr(config_class, "sample_run_config"):
        config: dict[str, Any] = config_class.sample_run_config()
        return config
    return {}


ENABLED_INFERENCE_PROVIDERS = [
    "ollama",
    "vllm",
    "tgi",
    "fireworks",
    "together",
    "gemini",
    "groq",
    "sambanova",
    "anthropic",
    "openai",
    "cerebras",
    "nvidia",
    "bedrock",
]

INFERENCE_PROVIDER_IDS = {
    "vllm": "${env.VLLM_URL:+vllm}",
    "tgi": "${env.TGI_URL:+tgi}",
    "cerebras": "${env.CEREBRAS_API_KEY:+cerebras}",
    "nvidia": "${env.NVIDIA_API_KEY:+nvidia}",
}


def get_remote_inference_providers() -> list[Provider]:
    # Filter out inline providers and some others - the starter distro only exposes remote providers
    remote_providers = [
        provider
        for provider in available_providers()
        if isinstance(provider, RemoteProviderSpec) and provider.adapter.adapter_type in ENABLED_INFERENCE_PROVIDERS
    ]

    inference_providers = []
    for provider_spec in remote_providers:
        provider_type = provider_spec.adapter.adapter_type

        if provider_type in INFERENCE_PROVIDER_IDS:
            provider_id = INFERENCE_PROVIDER_IDS[provider_type]
        else:
            provider_id = provider_type.replace("-", "_").replace("::", "_")
        config = _get_config_for_provider(provider_spec)

        inference_providers.append(
            Provider(
                provider_id=provider_id,
                provider_type=f"remote::{provider_type}",
                config=config,
            )
        )
    return inference_providers


def get_distribution_template() -> DistributionTemplate:
    remote_inference_providers = get_remote_inference_providers()
    name = "starter"

    providers = {
        "inference": [BuildProvider(provider_type=p.provider_type, module=p.module) for p in remote_inference_providers]
        + [BuildProvider(provider_type="inline::sentence-transformers")],
        "vector_io": [
            BuildProvider(provider_type="inline::faiss"),
            BuildProvider(provider_type="inline::sqlite-vec"),
            BuildProvider(provider_type="inline::milvus"),
            BuildProvider(provider_type="remote::chromadb"),
            BuildProvider(provider_type="remote::pgvector"),
        ],
        "files": [BuildProvider(provider_type="inline::localfs")],
        "safety": [BuildProvider(provider_type="inline::llama-guard")],
        "agents": [BuildProvider(provider_type="inline::meta-reference")],
        "telemetry": [BuildProvider(provider_type="inline::meta-reference")],
        "post_training": [BuildProvider(provider_type="inline::huggingface")],
        "eval": [BuildProvider(provider_type="inline::meta-reference")],
        "datasetio": [
            BuildProvider(provider_type="remote::huggingface"),
            BuildProvider(provider_type="inline::localfs"),
        ],
        "scoring": [
            BuildProvider(provider_type="inline::basic"),
            BuildProvider(provider_type="inline::llm-as-judge"),
            BuildProvider(provider_type="inline::braintrust"),
        ],
        "tool_runtime": [
            BuildProvider(provider_type="remote::brave-search"),
            BuildProvider(provider_type="remote::tavily-search"),
            BuildProvider(provider_type="inline::rag-runtime"),
            BuildProvider(provider_type="remote::model-context-protocol"),
        ],
    }
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
    default_shields = [
        # if the
        ShieldInput(
            shield_id="llama-guard",
            provider_id="${env.SAFETY_MODEL:+llama-guard}",
            provider_shield_id="${env.SAFETY_MODEL:=}",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Quick start template for running Llama Stack with several popular providers",
        container_image=None,
        template_path=None,
        providers=providers,
        additional_pip_packages=PostgresSqlStoreConfig.pip_packages(),
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": remote_inference_providers + [embedding_provider],
                    "vector_io": [
                        Provider(
                            provider_id="faiss",
                            provider_type="inline::faiss",
                            config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
                        ),
                        Provider(
                            provider_id="sqlite-vec",
                            provider_type="inline::sqlite-vec",
                            config=SQLiteVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
                        ),
                        Provider(
                            provider_id="${env.MILVUS_URL:+milvus}",
                            provider_type="inline::milvus",
                            config=MilvusVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
                        ),
                        Provider(
                            provider_id="${env.CHROMADB_URL:+chromadb}",
                            provider_type="remote::chromadb",
                            config=ChromaVectorIOConfig.sample_run_config(
                                f"~/.llama/distributions/{name}/",
                                url="${env.CHROMADB_URL:=}",
                            ),
                        ),
                        Provider(
                            provider_id="${env.PGVECTOR_DB:+pgvector}",
                            provider_type="remote::pgvector",
                            config=PGVectorVectorIOConfig.sample_run_config(
                                f"~/.llama/distributions/{name}",
                                db="${env.PGVECTOR_DB:=}",
                                user="${env.PGVECTOR_USER:=}",
                                password="${env.PGVECTOR_PASSWORD:=}",
                            ),
                        ),
                    ],
                    "files": [files_provider],
                },
                default_models=[],
                default_tool_groups=default_tool_groups,
                default_shields=default_shields,
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
        },
    )
