# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import Provider

from llama_stack.providers.inline.memory.faiss.config import FaissImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::bedrock"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["remote::bedrock"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
    }
    name = "bedrock"
    memory_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissImplConfig.sample_run_config(f"distributions/{name}"),
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use AWS Bedrock for running LLM inference and safety",
        docker_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        default_models=[],
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "memory": [memory_provider],
                },
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
        },
    )
