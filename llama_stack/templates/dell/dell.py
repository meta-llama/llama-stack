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
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": [
            Provider(provider_id="tgi", provider_type="remote::tgi"),
            Provider(provider_id="sentence-transformers", provider_type="inline::sentence-transformers"),
        ],
        "vector_io": [
            Provider(provider_id="faiss", provider_type="inline::faiss"),
            Provider(provider_id="chromadb", provider_type="remote::chromadb"),
            Provider(provider_id="pgvector", provider_type="remote::pgvector"),
        ],
        "safety": [Provider(provider_id="llama-guard", provider_type="inline::llama-guard")],
        "agents": [Provider(provider_id="meta-reference", provider_type="inline::meta-reference")],
        "telemetry": [Provider(provider_id="meta-reference", provider_type="inline::meta-reference")],
        "eval": [Provider(provider_id="meta-reference", provider_type="inline::meta-reference")],
        "datasetio": [
            Provider(provider_id="huggingface", provider_type="remote::huggingface"),
            Provider(provider_id="localfs", provider_type="inline::localfs"),
        ],
        "scoring": [
            Provider(provider_id="basic", provider_type="inline::basic"),
            Provider(provider_id="llm-as-judge", provider_type="inline::llm-as-judge"),
            Provider(provider_id="braintrust", provider_type="inline::braintrust"),
        ],
        "tool_runtime": [
            Provider(provider_id="brave-search", provider_type="remote::brave-search"),
            Provider(provider_id="tavily-search", provider_type="remote::tavily-search"),
            Provider(provider_id="rag-runtime", provider_type="inline::rag-runtime"),
        ],
    }
    name = "dell"
    inference_provider = Provider(
        provider_id="tgi0",
        provider_type="remote::tgi",
        config={
            "url": "${env.DEH_URL}",
        },
    )
    safety_inference_provider = Provider(
        provider_id="tgi1",
        provider_type="remote::tgi",
        config={
            "url": "${env.DEH_SAFETY_URL}",
        },
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )
    chromadb_provider = Provider(
        provider_id="chromadb",
        provider_type="remote::chromadb",
        config={
            "url": "${env.CHROMA_URL}",
        },
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="tgi0",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="tgi1",
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
            provider_id="brave-search",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::rag",
            provider_id="rag-runtime",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Dell's distribution of Llama Stack. TGI inference via Dell's custom container",
        container_image=None,
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider, embedding_provider],
                    "vector_io": [chromadb_provider],
                },
                default_models=[inference_model, embedding_model],
                default_tool_groups=default_tool_groups,
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [
                        inference_provider,
                        safety_inference_provider,
                        embedding_provider,
                    ],
                    "vector_io": [chromadb_provider],
                },
                default_models=[inference_model, safety_model, embedding_model],
                default_shields=[ShieldInput(shield_id="${env.SAFETY_MODEL}")],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "DEH_URL": (
                "http://0.0.0.0:8181",
                "URL for the Dell inference server",
            ),
            "DEH_SAFETY_URL": (
                "http://0.0.0.0:8282",
                "URL for the Dell safety inference server",
            ),
            "CHROMA_URL": (
                "http://localhost:6601",
                "URL for the Chroma server",
            ),
            "INFERENCE_MODEL": (
                "meta-llama/Llama-3.2-3B-Instruct",
                "Inference model loaded into the TGI server",
            ),
            "SAFETY_MODEL": (
                "meta-llama/Llama-Guard-3-1B",
                "Name of the safety (Llama-Guard) model to use",
            ),
        },
    )
