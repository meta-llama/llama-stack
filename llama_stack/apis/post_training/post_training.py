# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.common.training_types import Checkpoint
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


@json_schema_type
class OptimizerType(Enum):
    adam = "adam"
    adamw = "adamw"
    sgd = "sgd"


@json_schema_type
class DatasetFormat(Enum):
    instruct = "instruct"
    dialog = "dialog"


@json_schema_type
class DataConfig(BaseModel):
    dataset_id: str
    batch_size: int
    shuffle: bool
    data_format: DatasetFormat
    validation_dataset_id: Optional[str] = None
    packed: Optional[bool] = False
    train_on_input: Optional[bool] = False


@json_schema_type
class OptimizerConfig(BaseModel):
    optimizer_type: OptimizerType
    lr: float
    weight_decay: float
    num_warmup_steps: int


@json_schema_type
class EfficiencyConfig(BaseModel):
    enable_activation_checkpointing: Optional[bool] = False
    enable_activation_offloading: Optional[bool] = False
    memory_efficient_fsdp_wrap: Optional[bool] = False
    fsdp_cpu_offload: Optional[bool] = False


@json_schema_type
class TrainingConfig(BaseModel):
    n_epochs: int
    max_steps_per_epoch: int
    gradient_accumulation_steps: int
    max_validation_steps: int
    data_config: DataConfig
    optimizer_config: OptimizerConfig
    efficiency_config: Optional[EfficiencyConfig] = None
    dtype: Optional[str] = "bf16"


@json_schema_type
class LoraFinetuningConfig(BaseModel):
    type: Literal["LoRA"] = "LoRA"
    lora_attn_modules: List[str]
    apply_lora_to_mlp: bool
    apply_lora_to_output: bool
    rank: int
    alpha: int
    use_dora: Optional[bool] = False
    quantize_base: Optional[bool] = False


@json_schema_type
class QATFinetuningConfig(BaseModel):
    type: Literal["QAT"] = "QAT"
    quantizer_name: str
    group_size: int


AlgorithmConfig = Annotated[Union[LoraFinetuningConfig, QATFinetuningConfig], Field(discriminator="type")]
register_schema(AlgorithmConfig, name="AlgorithmConfig")


@json_schema_type
class PostTrainingJobLogStream(BaseModel):
    """Stream of logs from a finetuning job."""

    job_uuid: str
    log_lines: List[str]


@json_schema_type
class RLHFAlgorithm(Enum):
    dpo = "dpo"


@json_schema_type
class DPOAlignmentConfig(BaseModel):
    reward_scale: float
    reward_clip: float
    epsilon: float
    gamma: float


@json_schema_type
class PostTrainingRLHFRequest(BaseModel):
    """Request to finetune a model."""

    job_uuid: str

    finetuned_model: URL

    dataset_id: str
    validation_dataset_id: str

    algorithm: RLHFAlgorithm
    algorithm_config: DPOAlignmentConfig

    optimizer_config: OptimizerConfig
    training_config: TrainingConfig

    # TODO: define these
    hyperparam_search_config: Dict[str, Any]
    logger_config: Dict[str, Any]


class PostTrainingJob(BaseModel):
    job_uuid: str


@json_schema_type
class PostTrainingJobStatusResponse(BaseModel):
    """Status of a finetuning job."""

    job_uuid: str
    status: JobStatus

    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    resources_allocated: Optional[Dict[str, Any]] = None

    checkpoints: List[Checkpoint] = Field(default_factory=list)


class ListPostTrainingJobsResponse(BaseModel):
    data: List[PostTrainingJob]


@json_schema_type
class PostTrainingJobArtifactsResponse(BaseModel):
    """Artifacts of a finetuning job."""

    job_uuid: str
    checkpoints: List[Checkpoint] = Field(default_factory=list)

    # TODO(ashwin): metrics, evals


class PostTraining(Protocol):
    @webmethod(route="/post-training/supervised-fine-tune", method="POST")
    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        model: str = Field(
            default="Llama3.2-3B-Instruct",
            description="Model descriptor from `llama model list`",
        ),
        checkpoint_dir: Optional[str] = None,
        algorithm_config: Optional[AlgorithmConfig] = None,
    ) -> PostTrainingJob: ...

    @webmethod(route="/post-training/preference-optimize", method="POST")
    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob: ...

    @webmethod(route="/post-training/jobs", method="GET")
    async def get_training_jobs(self) -> ListPostTrainingJobsResponse: ...

    @webmethod(route="/post-training/job/status", method="GET")
    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse: ...

    @webmethod(route="/post-training/job/cancel", method="POST")
    async def cancel_training_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/post-training/job/artifacts", method="GET")
    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse: ...
