# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import List

from pydantic import BaseModel

from strong_typing.schema import json_schema_type


class OptimizerType(Enum):
    adam = "adam"
    adamw = "adamw"
    sgd = "sgd"


@json_schema_type
class OptimizerConfig(BaseModel):
    optimizer_type: OptimizerType
    lr: float
    lr_min: float
    weight_decay: float


@json_schema_type
class TrainingConfig(BaseModel):
    n_epochs: int
    batch_size: int
    shuffle: bool
    n_iters: int

    enable_activation_checkpointing: bool
    memory_efficient_fsdp_wrap: bool
    fsdp_cpu_offload: bool


@json_schema_type
class FinetuningAlgorithm(Enum):
    full = "full"
    lora = "lora"
    qlora = "qlora"
    dora = "dora"


@json_schema_type
class LoraFinetuningConfig(BaseModel):
    lora_attn_modules: List[str]
    apply_lora_to_mlp: bool
    apply_lora_to_output: bool
    rank: int
    alpha: int


@json_schema_type
class QLoraFinetuningConfig(LoraFinetuningConfig):
    pass


@json_schema_type
class DoraFinetuningConfig(LoraFinetuningConfig):
    pass


@json_schema_type
class PostTrainingJobLogStream(BaseModel):
    """Stream of logs from a finetuning job."""

    job_uuid: str
    log_lines: List[str]


@json_schema_type
class PostTrainingJobStatus(Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    scheduled = "scheduled"


@json_schema_type
class RLHFAlgorithm(Enum):
    dpo = "dpo"


@json_schema_type
class DPOAlignmentConfig(BaseModel):
    reward_scale: float
    reward_clip: float
    epsilon: float
    gamma: float
