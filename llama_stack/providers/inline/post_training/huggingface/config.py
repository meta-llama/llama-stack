# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Literal
import torch

from pydantic import BaseModel


class HuggingFacePostTrainingConfig(BaseModel):
    device: str = "cuda"
    distributed_backend: Literal["fsdp", "deepspeed"] | None = None
    checkpoint_format: Literal["full_state", "huggingface"] | None = "huggingface"
    chat_template: str = "<|user|>\n{input}\n<|assistant|>\n{output}"
    model_specific_config: dict = {
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "sdpa",
    }
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    save_total_limit: int = 3
    logging_steps: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True

    @classmethod
    def sample_run_config(cls, __distro_dir__: str, **kwargs: Any) -> dict[str, Any]:
        return {"checkpoint_format": "huggingface", "distributed_backend": None, "device": "cpu"}
