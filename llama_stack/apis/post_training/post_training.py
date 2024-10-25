# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum

from typing import Any, Dict, List, Optional, Protocol

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel, Field

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.datasets import *  # noqa: F403
from llama_stack.apis.common.training_types import *  # noqa: F403


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


@json_schema_type
class PostTrainingSFTRequest(BaseModel):
    """Request to finetune a model."""

    job_uuid: str

    model: str
    dataset_id: str
    validation_dataset_id: str

    algorithm: FinetuningAlgorithm
    algorithm_config: Union[
        LoraFinetuningConfig, QLoraFinetuningConfig, DoraFinetuningConfig
    ]

    optimizer_config: OptimizerConfig
    training_config: TrainingConfig

    # TODO: define these
    hyperparam_search_config: Dict[str, Any]
    logger_config: Dict[str, Any]


@json_schema_type
class PostTrainingRLHFRequest(BaseModel):
    """Request to finetune a model."""

    job_uuid: str

    finetuned_model: URL

    dataset_id: str
    validation_dataset_id: str

    algorithm: RLHFAlgorithm
    algorithm_config: Union[DPOAlignmentConfig]

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
    status: PostTrainingJobStatus

    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    resources_allocated: Optional[Dict[str, Any]] = None

    checkpoints: List[Checkpoint] = Field(default_factory=list)


@json_schema_type
class PostTrainingJobArtifactsResponse(BaseModel):
    """Artifacts of a finetuning job."""

    job_uuid: str
    checkpoints: List[Checkpoint] = Field(default_factory=list)

    # TODO(ashwin): metrics, evals


class PostTraining(Protocol):
    @webmethod(route="/post_training/supervised_fine_tune")
    def supervised_fine_tune(
        self,
        job_uuid: str,
        model: str,
        dataset_id: str,
        validation_dataset_id: str,
        algorithm: FinetuningAlgorithm,
        algorithm_config: Union[
            LoraFinetuningConfig, QLoraFinetuningConfig, DoraFinetuningConfig
        ],
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob: ...

    @webmethod(route="/post_training/preference_optimize")
    def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: URL,
        dataset_id: str,
        validation_dataset_id: str,
        algorithm: RLHFAlgorithm,
        algorithm_config: Union[DPOAlignmentConfig],
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob: ...

    @webmethod(route="/post_training/jobs")
    def get_training_jobs(self) -> List[PostTrainingJob]: ...

    # sends SSE stream of logs
    @webmethod(route="/post_training/job/logs")
    def get_training_job_logstream(self, job_uuid: str) -> PostTrainingJobLogStream: ...

    @webmethod(route="/post_training/job/status")
    def get_training_job_status(
        self, job_uuid: str
    ) -> PostTrainingJobStatusResponse: ...

    @webmethod(route="/post_training/job/cancel")
    def cancel_training_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/post_training/job/artifacts")
    def get_training_job_artifacts(
        self, job_uuid: str
    ) -> PostTrainingJobArtifactsResponse: ...
