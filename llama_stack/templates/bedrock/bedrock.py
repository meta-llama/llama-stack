# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["remote::bedrock"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["remote::bedrock"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
    }

    return DistributionTemplate(
        name="bedrock",
        distro_type="self_hosted",
        description="Use AWS Bedrock for running LLM inference and safety",
        docker_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        default_models=[],
        run_configs={
            "run.yaml": RunConfigSettings(),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
        },
    )
