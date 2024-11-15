# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field

from llama_stack.providers.utils.docker.service_config import DockerComposeServiceConfig


DEFAULT_VLLM_PORT = 8000


@json_schema_type
class VLLMInferenceAdapterConfig(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="The URL for the vLLM model serving endpoint",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate.",
    )
    api_token: Optional[str] = Field(
        default="fake",
        description="The API token",
    )

    @classmethod
    def sample_run_config(
        cls,
        url: str = "${env.VLLM_URL:http://host.docker.internal:5100/v1}",
    ):
        return {
            "url": url,
            "max_tokens": "${env.VLLM_MAX_TOKENS:4096}",
            "api_token": "${env.VLLM_API_TOKEN:fake}",
        }

    @classmethod
    def sample_docker_compose_config(
        cls,
        port: int = DEFAULT_VLLM_PORT,
        cuda_visible_devices: str = "0",
        model: str = "meta-llama/Llama-3.2-3B-Instruct",
    ) -> Optional[DockerComposeServiceConfig]:
        return DockerComposeServiceConfig(
            image="vllm/vllm-openai:latest",
            volumes=["$HOME/.cache/huggingface:/root/.cache/huggingface"],
            devices=["nvidia.com/gpu=all"],
            deploy={
                "resources": {
                    "reservations": {
                        "devices": [{"driver": "nvidia", "capabilities": ["gpu"]}]
                    }
                }
            },
            runtime="nvidia",
            ports=[f"{port}:{port}"],
            environment={
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
                "HUGGING_FACE_HUB_TOKEN": "$HF_TOKEN",
            },
            command=(
                " ".join(
                    [
                        "--gpu-memory-utilization 0.75",
                        f"--model {model}",
                        "--enforce-eager",
                        "--max-model-len 8192",
                        "--max-num-seqs 16",
                        f"--port {port}",
                    ]
                )
            ),
        )
