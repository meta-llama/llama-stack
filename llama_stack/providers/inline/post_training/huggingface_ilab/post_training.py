# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from datetime import datetime
from typing import Any, Dict, Optional

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    JobStatus,
    ListPostTrainingJobsResponse,
    LoraFinetuningConfig,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from llama_stack.providers.inline.post_training.huggingface_ilab.config import HFilabPostTrainingConfig
from llama_stack.schema_utils import webmethod


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

        # TODO: assume sync job, will need jobs API for async scheduling
        self.jobs = {}
        self.checkpoints_dict = {}

    async def shutdown(self):
        pass

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        model: str,
        checkpoint_dir: Optional[str],
        algorithm_config: Optional[AlgorithmConfig],
    ) -> PostTrainingJob:
        if job_uuid in self.jobs:
            raise ValueError(f"Job {job_uuid} already exists")

        post_training_job = PostTrainingJob(job_uuid=job_uuid)

        job_status_response = PostTrainingJobStatusResponse(
            job_uuid=job_uuid,
            status=JobStatus.scheduled,
            scheduled_at=datetime.now(),
        )
        self.jobs[job_uuid] = job_status_response

        if isinstance(algorithm_config, LoraFinetuningConfig):
            try:
                recipe = LoraFinetuningSingleDevice(
                    self.config,
                    job_uuid,
                    training_config,
                    hyperparam_search_config,
                    logger_config,
                    model,
                    checkpoint_dir,
                    algorithm_config,
                    self.datasetio_api,
                    self.datasets_api,
                )

                job_status_response.status = JobStatus.in_progress
                job_status_response.started_at = datetime.now()

                await recipe.setup()
                resources_allocated, checkpoints = await recipe.train()

                self.checkpoints_dict[job_uuid] = checkpoints
                job_status_response.resources_allocated = resources_allocated
                job_status_response.checkpoints = checkpoints
                job_status_response.status = JobStatus.completed
                job_status_response.completed_at = datetime.now()

            except Exception:
                job_status_response.status = JobStatus.failed
                raise
        else:
            raise NotImplementedError()

        return post_training_job

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
