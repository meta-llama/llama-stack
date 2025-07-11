# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.models import ModelType
from llama_stack.distribution.datatypes import ModelInput, Provider
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.inference.vllm import VLLMConfig
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.templates.template import (
    DistributionTemplate,
    RunConfigSettings,
    ToolGroupInput,
)


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": [
            Provider(
                provider_id="vllm",
                provider_type="inline::vllm",
                config={},
            ),
            Provider(
                provider_id="sentence-transformers",
                provider_type="inline::sentence-transformers",
                config={},
            ),
        ],
        "vector_io": [
            Provider(
                provider_id="faiss",
                provider_type="inline::faiss",
                config={},
            ),
            Provider(
                provider_id="chromadb",
                provider_type="remote::chromadb",
                config={},
            ),
            Provider(
                provider_id="pgvector",
                provider_type="remote::pgvector",
                config={},
            ),
        ],
        "safety": [
            Provider(
                provider_id="llama-guard",
                provider_type="inline::llama-guard",
                config={},
            )
        ],
        "agents": [
            Provider(
                provider_id="meta-reference",
                provider_type="inline::meta-reference",
                config={},
            )
        ],
        "telemetry": [
            Provider(
                provider_id="meta-reference",
                provider_type="inline::meta-reference",
                config={},
            )
        ],
        "eval": [
            Provider(
                provider_id="meta-reference",
                provider_type="inline::meta-reference",
                config={},
            )
        ],
        "datasetio": [
            Provider(
                provider_id="huggingface",
                provider_type="remote::huggingface",
                config={},
            ),
            Provider(
                provider_id="localfs",
                provider_type="inline::localfs",
                config={},
            ),
        ],
        "scoring": [
            Provider(
                provider_id="basic",
                provider_type="inline::basic",
                config={},
            ),
            Provider(
                provider_id="llm-as-judge",
                provider_type="inline::llm-as-judge",
                config={},
            ),
            Provider(
                provider_id="braintrust",
                provider_type="inline::braintrust",
                config={},
            ),
        ],
        "tool_runtime": [
            Provider(
                provider_id="brave-search",
                provider_type="remote::brave-search",
                config={},
            ),
            Provider(
                provider_id="tavily-search",
                provider_type="remote::tavily-search",
                config={},
            ),
            Provider(
                provider_id="rag-runtime",
                provider_type="inline::rag-runtime",
                config={},
            ),
            Provider(
                provider_id="model-context-protocol",
                provider_type="remote::model-context-protocol",
                config={},
            ),
        ],
    }

    name = "vllm-gpu"
    inference_provider = Provider(
        provider_id="vllm",
        provider_type="inline::vllm",
        config=VLLMConfig.sample_run_config(),
    )
    vector_io_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="vllm",
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
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use a built-in vLLM engine for running LLM inference",
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
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "8321",
                "Port for the Llama Stack distribution server",
            ),
            "INFERENCE_MODEL": (
                "meta-llama/Llama-3.2-3B-Instruct",
                "Inference model loaded into the vLLM engine",
            ),
            "TENSOR_PARALLEL_SIZE": (
                "1",
                "Number of tensor parallel replicas (number of GPUs to use).",
            ),
            "MAX_TOKENS": (
                "4096",
                "Maximum number of tokens to generate.",
            ),
            "ENFORCE_EAGER": (
                "False",
                "Whether to use eager mode for inference (otherwise cuda graphs are used).",
            ),
            "GPU_MEMORY_UTILIZATION": (
                "0.7",
                "GPU memory utilization for the vLLM engine.",
            ),
        },
    )
