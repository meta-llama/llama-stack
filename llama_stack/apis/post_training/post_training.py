# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Literal, Protocol

from pydantic import BaseModel, Field

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
    validation_dataset_id: str | None = None
    packed: bool | None = False
    train_on_input: bool | None = False


@json_schema_type
class OptimizerConfig(BaseModel):
    optimizer_type: OptimizerType
    lr: float
    weight_decay: float
    num_warmup_steps: int


@json_schema_type
class EfficiencyConfig(BaseModel):
    enable_activation_checkpointing: bool | None = False
    enable_activation_offloading: bool | None = False
    memory_efficient_fsdp_wrap: bool | None = False
    fsdp_cpu_offload: bool | None = False


@json_schema_type
class TrainingConfig(BaseModel):
    n_epochs: int
    max_steps_per_epoch: int = 1
    gradient_accumulation_steps: int = 1
    max_validation_steps: int | None = 1
    data_config: DataConfig | None = None
    optimizer_config: OptimizerConfig | None = None
    efficiency_config: EfficiencyConfig | None = None
    dtype: str | None = "bf16"


@json_schema_type
class LoraFinetuningConfig(BaseModel):
    type: Literal["LoRA"] = "LoRA"
    lora_attn_modules: list[str]
    apply_lora_to_mlp: bool
    apply_lora_to_output: bool
    rank: int
    alpha: int
    use_dora: bool | None = False
    quantize_base: bool | None = False


@json_schema_type
class QATFinetuningConfig(BaseModel):
    type: Literal["QAT"] = "QAT"
    quantizer_name: str
    group_size: int


AlgorithmConfig = Annotated[LoraFinetuningConfig | QATFinetuningConfig, Field(discriminator="type")]
register_schema(AlgorithmConfig, name="AlgorithmConfig")


@json_schema_type
class PostTrainingJobLogStream(BaseModel):
    """Stream of logs from a finetuning job."""

    job_uuid: str
    log_lines: list[str]


@json_schema_type
class RLHFAlgorithm(Enum):
    dpo = "dpo"


@json_schema_type
class DPOLossType(Enum):
    sigmoid = "sigmoid"
    hinge = "hinge"
    ipo = "ipo"
    kto_pair = "kto_pair"


@json_schema_type
class DPOAlignmentConfig(BaseModel):
    beta: float
    loss_type: DPOLossType = DPOLossType.sigmoid


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
    hyperparam_search_config: dict[str, Any]
    logger_config: dict[str, Any]


class PostTrainingJob(BaseModel):
    job_uuid: str


@json_schema_type
class PostTrainingJobStatusResponse(BaseModel):
    """Status of a finetuning job."""

    job_uuid: str
    status: JobStatus

    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    resources_allocated: dict[str, Any] | None = None

    checkpoints: list[Checkpoint] = Field(default_factory=list)


class ListPostTrainingJobsResponse(BaseModel):
    data: list[PostTrainingJob]


@json_schema_type
class PostTrainingJobArtifactsResponse(BaseModel):
    """Artifacts of a finetuning job."""

    job_uuid: str
    checkpoints: list[Checkpoint] = Field(default_factory=list)

    # TODO(ashwin): metrics, evals


class PostTraining(Protocol):
    @webmethod(route="/post-training/supervised-fine-tune", method="POST")
    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str | None = Field(
            default=None,
            description="Model descriptor for training if not in provider config`",
        ),
        checkpoint_dir: str | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> PostTrainingJob:
        """Run supervised fine-tuning of a model.

        :param job_uuid: The UUID of the job to create.
        :param training_config: The training configuration.
        :param hyperparam_search_config: The hyperparam search configuration.
        :param logger_config: The logger configuration.
        :param model: The model to fine-tune.
        :param checkpoint_dir: The directory to save checkpoint(s) to.
        :param algorithm_config: The algorithm configuration.
        :returns: A PostTrainingJob.
        """
        ...

    @webmethod(route="/post-training/preference-optimize", method="POST")
    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
    ) -> PostTrainingJob:
        """Run preference optimization of a model.

        :param job_uuid: The UUID of the job to create.
        :param finetuned_model: The model to fine-tune.
        :param algorithm_config: The algorithm configuration.
        :param training_config: The training configuration.
        :param hyperparam_search_config: The hyperparam search configuration.
        :param logger_config: The logger configuration.
        :returns: A PostTrainingJob.
        """
        ...

    @webmethod(route="/post-training/jobs", method="GET")
    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        """Get all training jobs.

        :returns: A ListPostTrainingJobsResponse.
        """
        ...

    @webmethod(route="/post-training/job/status", method="GET")
    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse:
        """Get the status of a training job.

        :param job_uuid: The UUID of the job to get the status of.
        :returns: A PostTrainingJobStatusResponse.
        """
        ...

    @webmethod(route="/post-training/job/cancel", method="POST")
    async def cancel_training_job(self, job_uuid: str) -> None:
        """Cancel a training job.

        :param job_uuid: The UUID of the job to cancel.
        """
        ...

    @webmethod(route="/post-training/job/artifacts", method="GET")
    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse:
        """Get the artifacts of a training job.

        :param job_uuid: The UUID of the job to get the artifacts of.
        :returns: A PostTrainingJobArtifactsResponse.
        """
        ...
