# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import (
    ModelInput,
    Provider,
    ShieldInput,
    ToolGroupInput,
)
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.remote.inference.podman_ai_lab import PodmanAILabImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::podman-ai-lab"],
        "vector_io": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
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
            "inline::rag-runtime",
            "remote::model-context-protocol",
            "remote::wolfram-alpha",
        ],
    }
    name = "podman-ai-lab"
    inference_provider = Provider(
        provider_id="podman-ai-lab",
        provider_type="remote::podman-ai-lab",
        config=PodmanAILabImplConfig.sample_run_config(),
    )
    vector_io_provider_faiss = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissVectorIOConfig.sample_run_config(f"~/.llama/distributions/{name}"),
    )

    safety_model = ModelInput(
        model_id="${env.SAFETY_MODEL}",
        provider_id="podman-ai-lab",
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
        ToolGroupInput(
            toolgroup_id="builtin::wolfram_alpha",
            provider_id="wolfram-alpha",
        ),
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use (an external) Podman AI Lab server for running LLM inference",
        container_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "vector_io": [vector_io_provider_faiss],
                },
                default_models=[],
                default_tool_groups=default_tool_groups,
            ),
            "run-with-safety.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "vector_io": [vector_io_provider_faiss],
                    "safety": [
                        Provider(
                            provider_id="llama-guard",
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
                    safety_model,
                ],
                default_shields=[
                    ShieldInput(
                        shield_id="${env.SAFETY_MODEL}",
                        provider_id="llama-guard",
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
            "PODMAN_AI_LAB_URL": (
                "http://127.0.0.1:10434",
                "URL of the Podman AI Lab server",
            ),
            "SAFETY_MODEL": (
                "meta-llama/Llama-Guard-3-1B",
                "Safety model loaded into the Podman AI Lab server",
            ),
        },
    )
