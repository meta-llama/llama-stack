# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from asyncio import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import fastapi
import fastapi.concurrency
import pydantic
from starlette.background import BackgroundTasks
from starlette.responses import JSONResponse

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    JobStatus,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from llama_stack.providers.inline.post_training.huggingface_ilab.config import HFilabPostTrainingConfig
from llama_stack.providers.inline.post_training.huggingface_ilab.recipes import FullPrecisionFineTuning
from llama_stack.schema_utils import webmethod


class TuningJob(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
    job_uuid: str
    status: list[JobStatus] = []

    created_at: datetime | None = None
    scheduled_at: datetime | None = None
    completed_at: datetime | None = None

    subproc_ref: subprocess.Process | None = None


class HFilabPostTrainingImpl:
    def __init__(
        self,
        config: HFilabPostTrainingConfig,
        datasetio_api: DatasetIO,
        datasets: Datasets,
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api
        self.datasets_api = datasets

        self.current_job: TuningJob | None = None

    async def shutdown(self):
        pass

    async def can_schedule_new_job(self) -> bool:
        if self.current_job is None:
            return True

        finalized_job_states = [JobStatus.completed, JobStatus.failed]

        # check most recent status of job.
        if self.current_job.status[-1] in finalized_job_states:
            return True

        return False

    def __set_status_callback(self, new_status: JobStatus):
        if self.current_job is not None:
            self.current_job.status.append(new_status)

    def __set_subproc_ref_callback(self, subproc_ref: subprocess.Process):
        if self.current_job is not None:
            self.current_job.subproc_ref = subproc_ref

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        model: str,
        checkpoint_dir: Optional[str],
        algorithm_config: Optional[AlgorithmConfig],
    ) -> JSONResponse:
        if not await self.can_schedule_new_job():
            # TODO: this status code isn't making its way up to the user. User just getting 500 from SDK.
            raise fastapi.HTTPException(
                status_code=503,  # service unavailable, try again later.
                detail="A tuning job is currently running; this could take a while.",
                headers={"Retry-After": "3600"},  # 60sec * 60min = 3600 seconds
            )

        recipe = FullPrecisionFineTuning(
            model=model,
            training_config=training_config,
            logger_config=logger_config,
            storage_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            algorithm_config=algorithm_config,
            datasets_api=self.datasets_api,
            datasetsio_api=self.datasetio_api,
        )

        # This is not a reliable or tidy way to implement the behavior that we want.
        tasks = BackgroundTasks()
        tasks.add_task(
            recipe.load_dataset_from_datasetsio,  # asynchronous request
        )
        tasks.add_task(
            recipe.preflight,  # synchronous request
            set_status_callback=self.__set_status_callback,
        )
        tasks.add_task(
            recipe.setup,  # synchronous request
        )
        tasks.add_task(
            recipe.train,  # asynchronous request
            set_status_callback=self.__set_status_callback,
            set_subproc_ref_callback=self.__set_subproc_ref_callback,
        )

        self.current_job = TuningJob(job_uuid=job_uuid, status=[JobStatus.scheduled])
        resp_object = PostTrainingJob(job_uuid=job_uuid)
        return JSONResponse(
            content=resp_object.model_dump(),
            background=tasks,
        )

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob:
        raise NotImplementedError("preference optimization is not implemented yet")

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        raise NotImplementedError("'get training jobs' ys not implemented yet")

    @webmethod(route="/post-training/job/status")  # type: ignore
    async def get_training_job_status(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]:
        raise NotImplementedError("'get training job status' is not implemented yet")

    @webmethod(route="/post-training/job/cancel")  # type: ignore
    async def cancel_training_job(self, job_uuid: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    @webmethod(route="/post-training/job/artifacts")  # type: ignore
    async def get_training_job_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]:
        raise NotImplementedError("'get training job artifacts' is not implemented yet")
