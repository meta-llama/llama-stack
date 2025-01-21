# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_models.sku_list import all_registered_models
from llama_stack.apis.models.models import ModelType
from llama_stack.distribution.datatypes import ModelInput, Provider, ShieldInput
from llama_stack.providers.inline.inference.sentence_transformers import (
    SentenceTransformersInferenceConfig,
)
from llama_stack.providers.inline.memory.faiss.config import FaissImplConfig
from llama_stack.providers.remote.inference.centml.config import (
    CentMLImplConfig,
)

# If your CentML adapter has a MODEL_ALIASES constant with known model mappings:
from llama_stack.providers.remote.inference.centml.centml import MODEL_ALIASES

from llama_stack.templates.template import (
    DistributionTemplate,
    RunConfigSettings,
)


def get_distribution_template() -> DistributionTemplate:
    """
    Returns a distribution template for running Llama Stack with CentML inference.
    """
    providers = {
        "inference": ["remote::centml"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": [
            "inline::basic",
            "inline::llm-as-judge",
            "inline::braintrust",
        ],
    }
    name = "centml"

    # Primary inference provider: CentML
    inference_provider = Provider(
        provider_id="centml",
        provider_type="remote::centml",
        config=CentMLImplConfig.sample_run_config(),
    )

    # Memory provider: Faiss
    memory_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissImplConfig.sample_run_config(f"distributions/{name}"),
    )

    # Embedding provider: SentenceTransformers
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )

    # Map Llama Models to provider IDs if needed
    core_model_to_hf_repo = {
        m.descriptor(): m.huggingface_repo for m in all_registered_models()
    }
    default_models = [
        ModelInput(
            model_id=core_model_to_hf_repo[m.llama_model],
            provider_model_id=m.provider_model_id,
            provider_id="centml",
        )
        for m in MODEL_ALIASES
    ]

    # Example embedding model
    embedding_model = ModelInput(
        model_id="all-MiniLM-L6-v2",
        provider_id="sentence-transformers",
        model_type=ModelType.embedding,
        metadata={"embedding_dimension": 384},
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use CentML for running LLM inference",
        docker_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        default_models=default_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider, embedding_provider],
                    "memory": [memory_provider],
                },
                default_models=default_models + [embedding_model],
                default_shields=[
                    ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")
                ],
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "CENTML_API_KEY": (
                "",
                "CentML API Key",
            ),
        },
    )
