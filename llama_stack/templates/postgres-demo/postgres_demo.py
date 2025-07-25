# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.apis.models import ModelType
from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.providers.inline.inference.sentence_transformers import SentenceTransformersInferenceConfig
from llama_stack.providers.remote.inference.vllm import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.utils.kvstore.config import PostgresKVStoreConfig
from llama_stack.providers.utils.sqlstore.sqlstore import PostgresSqlStoreConfig
from llama_stack.templates.template import (
    DistributionTemplate,
    RunConfigSettings,
)


def get_distribution_template() -> DistributionTemplate:
    inference_providers = [
        Provider(
            provider_id="vllm-inference",
            provider_type="remote::vllm",
            config=VLLMInferenceAdapterConfig.sample_run_config(
                url="${env.VLLM_URL:=http://localhost:8000/v1}",
            ),
        ),
    ]
    providers = {
        "inference": inference_providers
        + [
            Provider(provider_id="sentence-transformers", provider_type="inline::sentence-transformers"),
        ],
        "vector_io": [
            Provider(provider_id="chromadb", provider_type="remote::chromadb"),
        ],
        "safety": [Provider(provider_id="llama-guard", provider_type="inline::llama-guard")],
        "agents": [Provider(provider_id="meta-reference", provider_type="inline::meta-reference")],
        "telemetry": [Provider(provider_id="meta-reference", provider_type="inline::meta-reference")],
        "tool_runtime": [
            Provider(provider_id="brave-search", provider_type="remote::brave-search"),
            Provider(provider_id="tavily-search", provider_type="remote::tavily-search"),
            Provider(provider_id="rag-runtime", provider_type="inline::rag-runtime"),
            Provider(
                provider_id="model-context-protocol",
                provider_type="remote::model-context-protocol",
            ),
        ],
    }
    name = "postgres-demo"

    vector_io_providers = [
        Provider(
            provider_id="${env.ENABLE_CHROMADB:+chromadb}",
            provider_type="remote::chromadb",
            config=ChromaVectorIOConfig.sample_run_config(
                f"~/.llama/distributions/{name}",
                url="${env.CHROMADB_URL:=}",
            ),
        ),
    ]
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

    default_models = [
        ModelInput(
            model_id="${env.INFERENCE_MODEL}",
            provider_id="vllm-inference",
        )
    ]
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )
    embedding_model = ModelInput(
        model_id="all-MiniLM-L6-v2",
        provider_id=embedding_provider.provider_id,
        model_type=ModelType.embedding,
        metadata={
            "embedding_dimension": 384,
        },
    )
    postgres_config = PostgresSqlStoreConfig.sample_run_config()
    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Quick start template for running Llama Stack with several popular providers",
        container_image=None,
        template_path=None,
        providers=providers,
        available_models_by_provider={},
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": inference_providers + [embedding_provider],
                    "vector_io": vector_io_providers,
                    "agents": [
                        Provider(
                            provider_id="meta-reference",
                            provider_type="inline::meta-reference",
                            config=dict(
                                persistence_store=postgres_config,
                                responses_store=postgres_config,
                            ),
                        )
                    ],
                    "telemetry": [
                        Provider(
                            provider_id="meta-reference",
                            provider_type="inline::meta-reference",
                            config=dict(
                                service_name="${env.OTEL_SERVICE_NAME:=\u200b}",
                                sinks="${env.TELEMETRY_SINKS:=console,otel_trace}",
                                otel_trace_endpoint="${env.OTEL_TRACE_ENDPOINT:=http://localhost:4318/v1/traces}",
                            ),
                        )
                    ],
                },
                default_models=default_models + [embedding_model],
                default_tool_groups=default_tool_groups,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
                metadata_store=PostgresKVStoreConfig.sample_run_config(),
                inference_store=postgres_config,
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "8321",
                "Port for the Llama Stack distribution server",
            ),
        },
    )
