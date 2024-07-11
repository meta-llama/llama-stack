from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from model_types import Message, URL

from strong_typing.schema import json_schema_type


class DatasetColumnType(Enum):
    dialog = "dialog"
    text = "text"
    media = "media"
    number = "number"
    json = "json"


@json_schema_type
@dataclass
class Dataset:
    """Dataset to be used for training or evaluating language models."""

    # TODO(ashwin): figure out if we need to add an enum for a "dataset type"

    columns: Dict[str, DatasetColumnType]
    content_url: URL
    metadata: Dict[str, Any] = field(default_factory=dict)


class OptimizerType(Enum):
    adam = "adam"
    adamw = "adamw"
    sgd = "sgd"


@json_schema_type
@dataclass
class OptimizerConfig:
    optimizer_type: OptimizerType
    lr: float
    lr_min: float
    weight_decay: float


@json_schema_type
@dataclass
class TrainingConfig:
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
@dataclass
class LoraFinetuningConfig:
    lora_attn_modules: List[str]
    apply_lora_to_mlp: bool
    apply_lora_to_output: bool
    rank: int
    alpha: int


@dataclass
class QLoraFinetuningConfig(LoraFinetuningConfig):
    pass


@dataclass
class DoraFinetuningConfig(LoraFinetuningConfig):
    pass


@json_schema_type
@dataclass
class FinetuningJobLogStream:
    """Stream of logs from a finetuning job."""

    job_uuid: str
    log_lines: List[str]


class FinetuningJobStatus(Enum):
    running = "running"
    completed = "completed"
    failed = "failed"
    scheduled = "scheduled"
