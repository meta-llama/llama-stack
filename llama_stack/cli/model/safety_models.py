# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from llama_models.datatypes import *  # noqa: F403
from llama_models.sku_list import LlamaDownloadInfo


class PromptGuardModel(BaseModel):
    """Make a 'fake' Model-like object for Prompt Guard. Eventually this will be removed."""

    model_id: str = "Prompt-Guard-86M"
    description: str = (
        "Prompt Guard. NOTE: this model will not be provided via `llama` CLI soon."
    )
    is_featured: bool = False
    huggingface_repo: str = "meta-llama/Prompt-Guard-86M"
    max_seq_length: int = 2048
    is_instruct_model: bool = False
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )
    arch_args: Dict[str, Any] = Field(default_factory=dict)
    recommended_sampling_params: Optional[SamplingParams] = None

    def descriptor(self) -> str:
        return self.model_id

    model_config = ConfigDict(protected_namespaces=())


def prompt_guard_model_sku():
    return PromptGuardModel()


def prompt_guard_download_info():
    return LlamaDownloadInfo(
        folder="Prompt-Guard",
        files=[
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        pth_size=1,
    )
