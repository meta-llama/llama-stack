# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from llama_stack.distribution.datatypes import ModelInput, Provider
from llama_stack.providers.inline.inference.meta_reference import (
    MetaReferenceQuantizedInferenceConfig,
)
from llama_stack.providers.inline.memory.faiss.config import FaissImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["inline::meta-reference-quantized"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
    }
    name = "meta-reference-quantized-gpu"
    inference_provider = Provider(
        provider_id="meta-reference-inference",
        provider_type="inline::meta-reference-quantized",
        config=MetaReferenceQuantizedInferenceConfig.sample_run_config(
            model="${env.INFERENCE_MODEL}",
            checkpoint_dir="${env.INFERENCE_CHECKPOINT_DIR:null}",
        ),
    )
    memory_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissImplConfig.sample_run_config(f"distributions/{name}"),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="meta-reference-inference",
    )
    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use Meta Reference with fp8, int4 quantization for running LLM inference",
        template_path=Path(__file__).parent / "doc_template.md",
        providers=providers,
        default_models=[inference_model],
        run_configs={
            "run.yaml": RunConfigSettings(
                provider_overrides={
                    "inference": [inference_provider],
                    "memory": [memory_provider],
                },
                default_models=[inference_model],
            ),
        },
        run_config_env_vars={
            "LLAMASTACK_PORT": (
                "5001",
                "Port for the Llama Stack distribution server",
            ),
            "INFERENCE_MODEL": (
                "meta-llama/Llama-3.2-3B-Instruct",
                "Inference model loaded into the Meta Reference server",
            ),
            "INFERENCE_CHECKPOINT_DIR": (
                "null",
                "Directory containing the Meta Reference model checkpoint",
            ),
        },
    )
