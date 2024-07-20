from enum import Enum
from typing import Any, Dict, List

from models.llama3_1.api.datatypes import URL
from pydantic import BaseModel, Field

from strong_typing.schema import json_schema_type


class TrainEvalDatasetColumnType(Enum):
    dialog = "dialog"
    text = "text"
    media = "media"
    number = "number"
    json = "json"


@json_schema_type
class TrainEvalDataset(BaseModel):
    """Dataset to be used for training or evaluating language models."""

    # TODO(ashwin): figure out if we need to add an enum for a "dataset type"

    columns: Dict[str, TrainEvalDatasetColumnType]
    content_url: URL
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


class PostTrainingJobStatus(Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    scheduled = "scheduled"


class RLHFAlgorithm(Enum):
    dpo = "dpo"


@json_schema_type
class DPOAlignmentConfig(BaseModel):
    reward_scale: float
    reward_clip: float
    epsilon: float
    gamma: float
