# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.distribution.datatypes import ModelInput, Provider
from llama_stack.providers.inline.inference.vllm import VLLMConfig
from llama_stack.providers.inline.memory.faiss.config import FaissImplConfig
from llama_stack.templates.template import DistributionTemplate, RunConfigSettings


def get_distribution_template() -> DistributionTemplate:
    providers = {
        "inference": ["inline::vllm"],
        "memory": ["inline::faiss", "remote::chromadb", "remote::pgvector"],
        "safety": ["inline::llama-guard"],
        "agents": ["inline::meta-reference"],
        "telemetry": ["inline::meta-reference"],
        "eval": ["inline::meta-reference"],
        "datasetio": ["remote::huggingface", "inline::localfs"],
        "scoring": ["inline::basic", "inline::llm-as-judge", "inline::braintrust"],
    }
    name = "vllm-gpu"
    inference_provider = Provider(
        provider_id="vllm",
        provider_type="inline::vllm",
        config=VLLMConfig.sample_run_config(),
    )
    memory_provider = Provider(
        provider_id="faiss",
        provider_type="inline::faiss",
        config=FaissImplConfig.sample_run_config(f"distributions/{name}"),
    )

    inference_model = ModelInput(
        model_id="${env.INFERENCE_MODEL}",
        provider_id="vllm",
    )

    return DistributionTemplate(
        name=name,
        distro_type="self_hosted",
        description="Use a built-in vLLM engine for running LLM inference",
        docker_image=None,
        template_path=None,
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
