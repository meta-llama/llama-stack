# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type


@json_schema_type
class VLLMConfig(BaseModel):
    """Configuration for the vLLM inference provider.

    Note that the model name is no longer part of this static configuration.
    You can bind an instance of this provider to a specific model with the
    ``models.register()`` API call."""

    tensor_parallel_size: int = Field(
        default=1,
        description="Number of tensor parallel replicas (number of GPUs to use).",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum number of tokens to generate.",
    )
    max_model_len: int = Field(default=4096, description="Maximum context length to use during serving.")
    max_num_seqs: int = Field(default=4, description="Maximum parallel batch size for generation.")
    enforce_eager: bool = Field(
        default=False,
        description="Whether to use eager mode for inference (otherwise cuda graphs are used).",
    )
    gpu_memory_utilization: float = Field(
        default=0.3,
        description=(
            "How much GPU memory will be allocated when this provider has finished "
            "loading, including memory that was already allocated before loading."
        ),
    )

    @classmethod
    def sample_run_config(cls, **kwargs: Any) -> Dict[str, Any]:
        return {
            "tensor_parallel_size": "${env.TENSOR_PARALLEL_SIZE:1}",
            "max_tokens": "${env.MAX_TOKENS:4096}",
            "max_model_len": "${env.MAX_MODEL_LEN:4096}",
            "max_num_seqs": "${env.MAX_NUM_SEQS:4}",
            "enforce_eager": "${env.ENFORCE_EAGER:False}",
            "gpu_memory_utilization": "${env.GPU_MEMORY_UTILIZATION:0.3}",
        }
