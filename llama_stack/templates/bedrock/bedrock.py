# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_models.sku_list import all_registered_models

from llama_stack.apis.models import ModelInput
from llama_stack.distribution.datatypes import Provider

from llama_stack.providers.inline.memory.faiss.config import FaissImplConfig
from llama_stack.providers.remote.inference.bedrock.bedrock import MODEL_ALIASES
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

    core_model_to_hf_repo = {
        m.descriptor(): m.huggingface_repo for m in all_registered_models()
    }

    default_models = [
        ModelInput(
            model_id=core_model_to_hf_repo[m.llama_model],
            provider_model_id=m.provider_model_id,
            provider_id="bedrock",
        )
        for m in MODEL_ALIASES
    ]

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use AWS Bedrock for running LLM inference and safety",
        docker_image=None,
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        default_models=default_models,
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "memory": [memory_provider],
                },
                default_models=default_models,
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
        },
    )
