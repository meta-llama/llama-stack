# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, Optional

from llama_stack.apis.common.job_types import JobArtifact, JobStatus
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    ListPostTrainingJobsResponse,
    LoraFinetuningConfig,
    PostTrainingJob,
    TrainingConfig,
)
from llama_stack.providers.inline.post_training.torchtune.config import (
    TorchtunePostTrainingConfig,
)
from llama_stack.providers.inline.post_training.torchtune.recipes.lora_finetuning_single_device import (
    LoraFinetuningSingleDevice,
)
from llama_stack.schema_utils import webmethod


class TorchtunePostTrainingImpl:
    def __init__(
        self,
        config: TorchtunePostTrainingConfig,
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

        post_training_job = PostTrainingJob(id=job_uuid)
        post_training_job.update_status(JobStatus.scheduled)
        self.jobs[job_uuid] = post_training_job

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

                post_training_job.update_status(JobStatus.running)

                await recipe.setup()
                resources_allocated, checkpoints = await recipe.train()

                self.checkpoints_dict[job_uuid] = checkpoints
                post_training_job.artifacts = [
                    JobArtifact(name="resources", type="resources", metadata=resources_allocated),
                ] + checkpoints
                post_training_job.update_status(JobStatus.completed)

            except Exception:
                post_training_job.update_status(JobStatus.failed)
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
    ) -> PostTrainingJob: ...

    # TODO: should these be under post-training/supervised-fine-tune/?
    # CRUD operations on running jobs
    @webmethod(route="/post-training/jobs/{job_id:path}", method="GET")
    async def get_post_training_job(self, job_id: str) -> PostTrainingJob:
        return self.jobs[job_id]

    @webmethod(route="/post-training/jobs", method="GET")
    async def list_post_training_jobs(self) -> ListPostTrainingJobsResponse:
        return ListPostTrainingJobsResponse(data=list(self.jobs.values()))

    @webmethod(route="/post-training/jobs/{job_id:path}", method="POST")
    async def update_post_training_job(self, job: PostTrainingJob) -> PostTrainingJob:
        raise NotImplementedError

    @webmethod(route="/post-training/job/{job_id:path}", method="DELETE")
    async def delete_post_training_job(self, job_id: str) -> None:
        raise NotImplementedError

    # Note: pause/resume/cancel are achieved as follows:
    # - POST with status=paused
    # - POST with status=resuming
    # - POST with status=cancelled
