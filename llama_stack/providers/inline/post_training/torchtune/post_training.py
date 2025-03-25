# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from enum import Enum
from typing import Any, Dict, Optional

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
from llama_stack.providers.utils.scheduler import JobArtifact, Scheduler
from llama_stack.providers.utils.scheduler import JobStatus as SchedulerJobStatus
from llama_stack.schema_utils import webmethod


class TrainingArtifactType(Enum):
    CHECKPOINT = "checkpoint"
    RESOURCES_STATS = "resources_stats"


_JOB_TYPE_SUPERVISED_FINE_TUNE = "supervised-fine-tune"


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
        self._scheduler = Scheduler()

    async def shutdown(self) -> None:
        await self._scheduler.shutdown()

    @staticmethod
    def _resources_stats_to_artifact(resources_stats: Dict[str, Any]) -> JobArtifact:
        return JobArtifact(
            type=TrainingArtifactType.RESOURCES_STATS.value,
            name=TrainingArtifactType.RESOURCES_STATS.value,
            metadata=resources_stats,
        )

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
        if isinstance(algorithm_config, LoraFinetuningConfig):

            async def handler(on_log_message_cb, on_status_change_cb, on_artifact_collected_cb):
                on_log_message_cb("Starting Lora finetuning")

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

                await recipe.setup()

                resources_allocated, checkpoints = await recipe.train()

                on_artifact_collected_cb(self._resources_stats_to_artifact(resources_allocated))
                for checkpoint in checkpoints:
                    on_artifact_collected_cb(checkpoint)

                on_status_change_cb(SchedulerJobStatus.completed)
                on_log_message_cb("Lora finetuning completed")
        else:
            raise NotImplementedError()

        job_uuid = self._scheduler.schedule(_JOB_TYPE_SUPERVISED_FINE_TUNE, job_uuid, handler)

        # TODO: initialize with more data from scheduler
        return PostTrainingJob(job_uuid=job_uuid)

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
        # TODO: implement
        raise NotImplementedError

    @webmethod(route="/post-training/jobs", method="GET")
    async def list_post_training_jobs(self) -> ListPostTrainingJobsResponse:
        # TODO: populate other data
        return ListPostTrainingJobsResponse(
            data=[PostTrainingJob(job_uuid=job.id) for job in self._scheduler.get_jobs()]
        )

    @webmethod(route="/post-training/jobs/{job_id:path}", method="POST")
    async def update_post_training_job(self, job: PostTrainingJob) -> PostTrainingJob:
        # TODO: implement
        raise NotImplementedError

    @webmethod(route="/post-training/job/{job_id:path}", method="DELETE")
    async def delete_post_training_job(self, job_id: str) -> None:
        # TODO: implement
        raise NotImplementedError

    # Note: pause/resume/cancel are achieved as follows:
    # - POST with status=paused
    # - POST with status=resuming
    # - POST with status=cancelled
