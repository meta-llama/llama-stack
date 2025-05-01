# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llama_stack.models.llama.sku_list import LlamaDownloadInfo
from llama_stack.models.llama.sku_types import CheckpointQuantizationFormat


class PromptGuardModel(BaseModel):
    """Make a 'fake' Model-like object for Prompt Guard. Eventually this will be removed."""

    model_id: str
    huggingface_repo: str
    description: str = "Prompt Guard. NOTE: this model will not be provided via `llama` CLI soon."
    is_featured: bool = False
    max_seq_length: int = 512
    is_instruct_model: bool = False
    quantization_format: CheckpointQuantizationFormat = CheckpointQuantizationFormat.bf16
    arch_args: dict[str, Any] = Field(default_factory=dict)

    def descriptor(self) -> str:
        return self.model_id

    model_config = ConfigDict(protected_namespaces=())


def prompt_guard_model_skus():
    return [
        PromptGuardModel(model_id="Prompt-Guard-86M", huggingface_repo="meta-llama/Prompt-Guard-86M"),
        PromptGuardModel(
            model_id="Llama-Prompt-Guard-2-86M",
            huggingface_repo="meta-llama/Llama-Prompt-Guard-2-86M",
        ),
        PromptGuardModel(
            model_id="Llama-Prompt-Guard-2-22M",
            huggingface_repo="meta-llama/Llama-Prompt-Guard-2-22M",
        ),
    ]


def prompt_guard_model_sku_map() -> dict[str, Any]:
    return {model.model_id: model for model in prompt_guard_model_skus()}


def prompt_guard_download_info_map() -> dict[str, LlamaDownloadInfo]:
    return {
        model.model_id: LlamaDownloadInfo(
            folder="Prompt-Guard" if model.model_id == "Prompt-Guard-86M" else model.model_id,
            files=[
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
            pth_size=1,
        )
        for model in prompt_guard_model_skus()
    }
