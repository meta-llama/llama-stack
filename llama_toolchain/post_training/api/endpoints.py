# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

from datetime import datetime

from typing import Any, Dict, List, Optional, Protocol

from pydantic import BaseModel, Field

from pyopenapi import webmethod
from strong_typing.schema import json_schema_type

from llama_models.llama3_1.api.datatypes import *  # noqa: F403
from llama_toolchain.dataset.api.datatypes import *  # noqa: F403
from llama_toolchain.common.training_types import *  # noqa: F403
from .datatypes import *  # noqa: F403


@json_schema_type
class PostTrainingSFTRequest(BaseModel):
    """Request to finetune a model."""

    job_uuid: str

    model: PretrainedModel
    dataset: TrainEvalDataset
    validation_dataset: TrainEvalDataset

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

    dataset: TrainEvalDataset
    validation_dataset: TrainEvalDataset

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
    def post_supervised_fine_tune(
        self,
        request: PostTrainingSFTRequest,
    ) -> PostTrainingJob: ...

    @webmethod(route="/post_training/preference_optimize")
    def post_preference_optimize(
        self,
        request: PostTrainingRLHFRequest,
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
