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
class TrainingStrategy(BaseModel):
    # params that control Optimizer
    lr: Optional[Union[float, Literal["auto"]]] = "auto" 
    weight_decay: Optional[float] = 0.1
    num_warmup_steps: Optional[Union[int, Literal["auto"]]] = "auto" 
    
    # paramas that control how data is fed for training
    batch_size: Optional[Union[int, Literal["auto"]]] = "auto" 
    shuffle: Optional[bool] = True
    n_epochs: Optional[int] = 3
    
    # training loop control params
    max_training_steps: Optional[int] = None
    max_validation_steps: Optional[int] = None
    gradient_accumulation_steps: Optional[Union[int, Literal["auto"]]] = "auto" 
    
    # precision for training
    dtype: Optional[str] = "bf16"


@json_schema_type
class LoraFinetuningStrategy(BaseModel):
    type: Literal["LoRA"] = "LoRA"
    lora_attn_modules: Optional[List[str]] = ["q_proj", "v_proj", "output_proj"]
    apply_lora_to_mlp: Optional[bool] = True
    apply_lora_to_output: Optional[bool] = False
    rank: Optional[int] = 8
    alpha: Optional[int] = 16
    use_dora: Optional[bool] = False
    quantize_base: Optional[bool] = False


@json_schema_type
class QATFinetuningStrategy(BaseModel):
    type: Literal["QAT"] = "QAT"
    quantizer_name: str
    group_size: int


AlgorithmStrategy = register_schema(
    Annotated[Union[LoraFinetuningStrategy, QATFinetuningStrategy], Field(discriminator="type")],
    name="AlgorithmStrategy",
)


@json_schema_type
class PostTrainingJobLogStream(BaseModel):
    """Stream of logs from a finetuning job."""

    job_uuid: str
    log_lines: List[str]


@json_schema_type
class RLHFAlgorithm(Enum):
    dpo = "dpo"


@json_schema_type
class DPOAlignmentStrategy(BaseModel):
    reward_scale: float
    reward_clip: float
    epsilon: float
    gamma: float


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
        training_dataset_id: str,
        model: str = Field(
            default="Llama3.2-3B-Instruct",
            description="Model descriptor from `llama model list`",
        ),

        # Optional
        validation_dataset_id: Optional[str] = None,
        training_strategy: Optional[TrainingStrategy] = TrainingStrategy(),
        althorighm: Optional[AlgorithmStrategy] = LoraFinetuningStrategy(),
    ) -> PostTrainingJob: ...

    @webmethod(route="/post-training/preference-optimize", method="POST")
    async def preference_optimize(
        self,
        job_uuid: str,
        training_dataset_id: str,
        model: str = Field(
            default="Llama3.2-3B-Instruct",
            description="Model descriptor from `llama model list`",
        ),

        # Optional
        validation_dataset_id: Optional[str] = None,
        training_strategy: Optional[TrainingStrategy] = TrainingStrategy(),
        althorighm: Optional[AlgorithmStrategy] = LoraFinetuningStrategy(),
    ) -> PostTrainingJob: ...

    @webmethod(route="/post-training/jobs", method="GET")
    async def get_training_jobs(self) -> ListPostTrainingJobsResponse: ...

    @webmethod(route="/post-training/job/status", method="GET")
    async def get_training_job_status(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]: ...

    @webmethod(route="/post-training/job/cancel", method="POST")
    async def cancel_training_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/post-training/job/artifacts", method="GET")
    async def get_training_job_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]: ...
