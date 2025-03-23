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
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.remote.inference.tgi import InferenceAPIImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::hf::serverless", "inline::sentence-transformers"],
        "vector_io": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "tool_runtime": [
            "remote::brave-search",
            "remote::tavily-search",
            "inline::code-interpreter",
            "inline::rag-runtime",
            "remote::model-context-protocol",
        ],
    }

    name = "hf-serverless"
    inference_provider = Provider(
        provider_id="hf-serverless",
        provider_type="remote::hf::serverless",
        config=InferenceAPIImplConfig.sample_run_config(),
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )
    vector_io_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="hf-serverless",
    )
    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="hf-serverless-safety",
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
            toolgroup_id="builtin::code_interpreter",
            provider_id="code-interpreter",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use (an external) Hugging Face Inference Endpoint for running LLM inference",
        container_image=None,
        template_path=None,
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider, embedding_provider],
                    "vector_io": [vector_io_provider],
                },
                default_models=[inference_model, embedding_model],
                default_tool_groups=default_tool_groups,
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [
                        inference_provider,
                        embedding_provider,
                        Provider(
                            provider_id="hf-serverless-safety",
                            provider_type="remote::hf::serverless",
                            config=InferenceAPIImplConfig.sample_run_config(
                                repo="${env.SAFETY_MODEL}",
                            ),
                        ),
                    ],
                    "vector_io": [vector_io_provider],
                },
                default_models=[
                    inference_model,
                    safety_model,
                    embedding_model,
                ],
                default_shields=[ShieldInput(shield_id="${env.SAFETY_MODEL}")],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "8321",
                "Port for the Llama Stack distribution server",
            ),
            "HF_API_TOKEN": (
                "hf_...",
                "Hugging Face API token",
            ),
            "INFERENCE_MODEL": (
                "meta-llama/Llama-3.2-3B-Instruct",
                "Inference model to be served by the HF Serverless endpoint",
            ),
            "SAFETY_MODEL": (
                "meta-llama/Llama-Guard-3-1B",
                "Safety model to be served by the HF Serverless endpoint",
            ),
        },
    )
