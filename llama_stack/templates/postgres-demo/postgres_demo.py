# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.providers.remote.inference.fireworks.config import FireworksImplConfig
from llama_stack.providers.remote.inference.fireworks.models import (
    MODEL_ENTRIES as FIREWORKS_MODEL_ENTRIES,
)
from llama_stack.providers.remote.inference.vllm import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.vector_io.chroma.config import ChromaVectorIOConfig
from llama_stack.providers.utils.inference.model_registry import ProviderModelEntry
from llama_stack.providers.utils.kvstore.config import PostgresKVStoreConfig
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
            "fireworks",
            FIREWORKS_MODEL_ENTRIES,
            FireworksImplConfig.sample_run_config(api_key="${env.FIREWORKS_API_KEY:}"),
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
    inference_providers.append(
        Provider(
            provider_id="vllm-inference",
            provider_type="remote::vllm",
            config=VLLMInferenceAdapterConfig.sample_run_config(
                url="${env.VLLM_URL:http://localhost:8000/v1}",
            ),
        )
    )
    return inference_providers, available_models


def get_distribution_template() -> DistributionTemplate:
    inference_providers, available_models = get_inference_providers()
    providers = {
        "inference": ([p.provider_type for p in inference_providers]),
        "vector_io": ["remote::chromadb"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "tool_runtime": [
            "remote::brave-search",
            "remote::tavily-search",
            "inline::rag-runtime",
            "remote::model-context-protocol",
        ],
    }
    name = "postgres-demo"

    vector_io_providers = [
        Provider(
            provider_id="${env.ENABLE_CHROMADB+chromadb}",
            provider_type="remote::chromadb",
            config=ChromaVectorIOConfig.sample_run_config(url="${env.CHROMADB_URL:}"),
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

    default_models = get_model_registry(available_models)
    default_models.append(
        ModelInput(
            model_id="${env.INFERENCE_MODEL}",
            provider_id="vllm-inference",
        )
    )
    postgres_config = {
        "type": "postgres",
        "host": "${env.POSTGRES_HOST:localhost}",
        "port": "${env.POSTGRES_PORT:5432}",
        "db": "${env.POSTGRES_DB:llamastack}",
        "user": "${env.POSTGRES_USER:llamastack}",
        "password": "${env.POSTGRES_PASSWORD:llamastack}",
    }

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Quick start template for running Llama Stack with several popular providers",
        container_image=None,
        template_path=None,
        providers=providers,
        available_models_by_provider=available_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": inference_providers,
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
                                service_name="${env.OTEL_SERVICE_NAME:}",
                                sinks="${env.TELEMETRY_SINKS:console}",
                            ),
                        )
                    ],
                },
                default_models=default_models,
                default_tool_groups=default_tool_groups,
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
                metadata_store=PostgresKVStoreConfig.model_validate(postgres_config),
                inference_store=PostgresSqlStoreConfig.model_validate(postgres_config),
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
        },
    )
