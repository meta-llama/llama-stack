# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_models.sku_list import all_registered_models

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
from llama_stack.providers.inline.memory.faiss.config import FaissImplConfig
from llama_stack.providers.remote.inference.fireworks import FireworksImplConfig
from llama_stack.providers.remote.inference.fireworks.fireworks import MODEL_ALIASES
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::fireworks"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
        "tool_runtime": [
            "remote::brave-search",
            "remote::tavily-search",
            "inline::code-interpreter",
            "inline::memory-runtime",
        ],
    }

    name = "fireworks"

    inference_provider = Provider(
        provider_id="fireworks",
        provider_type="remote::fireworks",
        config=FireworksImplConfig.sample_run_config(),
    )
    embedding_provider = Provider(
        provider_id="sentence-transformers",
        provider_type="inline::sentence-transformers",
        config=SentenceTransformersInferenceConfig.sample_run_config(),
    )
    memory_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissImplConfig.sample_run_config(f"distributions/{name}"),
    )

    core_model_to_hf_repo = {
        m.descriptor(): m.huggingface_repo for m in all_registered_models()
    }
    default_models = [
        ModelInput(
            model_id=core_model_to_hf_repo[m.llama_model],
            provider_model_id=m.provider_model_id,
            provider_id="fireworks",
        )
        for m in MODEL_ALIASES
    ]
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
            toolgroup_id="builtin::memory",
            provider_id="memory-runtime",
        ),
        ToolGroupInput(
            toolgroup_id="builtin::code_interpreter",
            provider_id="code-interpreter",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use Fireworks.AI for running LLM inference",
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
                default_shields=[ShieldInput(shield_id="meta-llama/Llama-Guard-3-8B")],
                default_tool_groups=default_tool_groups,
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [
                        inference_provider,
                        embedding_provider,
                    ],
                    "memory": [memory_provider],
                    "safety": [
                        Provider(
                            provider_id="llama-guard",
                            provider_type="inline::llama-guard",
                            config={},
                        ),
                        Provider(
                            provider_id="llama-guard-vision",
                            provider_type="inline::llama-guard",
                            config={},
                        ),
                        Provider(
                            provider_id="code-scanner",
                            provider_type="inline::code-scanner",
                            config={},
                        ),
                    ],
                },
                default_models=[
                    *default_models,
                    embedding_model,
                ],
                default_shields=[
                    ShieldInput(
                        shield_id="meta-llama/Llama-Guard-3-8B",
                        provider_id="llama-guard",
                    ),
                    ShieldInput(
                        shield_id="meta-llama/Llama-Guard-3-11B-Vision",
                        provider_id="llama-guard-vision",
                    ),
                    ShieldInput(
                        shield_id="CodeScanner",
                        provider_id="code-scanner",
                    ),
                ],
                default_tool_groups=default_tool_groups,
            ),
        },
        run_config_env_vars={
            "LLAMA_STACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "FIREWORKS_API_KEY": (
                "",
                "Fireworks.AI API Key",
            ),
        },
    )
