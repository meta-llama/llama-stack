# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Any, Dict, Optional

from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    Checkpoint,
    DPOAlignmentConfig,
    JobStatus,
    ListPostTrainingJobsResponse,
    LoraFinetuningConfig,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
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

_ARTIFACT_TYPE_CHECKPOINT = "checkpoint"
_ARTIFACT_TYPE_RESOURCES_STATS = "resources_stats"

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
    def _checkpoint_to_artifact(checkpoint: Checkpoint) -> JobArtifact:
        return JobArtifact(
            type=_ARTIFACT_TYPE_CHECKPOINT,
            name=checkpoint.identifier,
            uri=checkpoint.path,
            metadata=dict(checkpoint),
        )

    @staticmethod
    def _resources_stats_to_artifact(resources_stats: Dict[str, Any]) -> JobArtifact:
        return JobArtifact(
            type=_ARTIFACT_TYPE_RESOURCES_STATS,
            name="resources_stats",
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
                    artifact = self._checkpoint_to_artifact(checkpoint)
                    on_artifact_collected_cb(artifact)

                on_status_change_cb(SchedulerJobStatus.completed)
                on_log_message_cb("Lora finetuning completed")
        else:
            raise NotImplementedError()

        job_uuid = self._scheduler.schedule(_JOB_TYPE_SUPERVISED_FINE_TUNE, job_uuid, handler)
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

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        return ListPostTrainingJobsResponse(
            data=[PostTrainingJob(job_uuid=job.id) for job in self._scheduler.get_jobs()]
        )

    @staticmethod
    def _get_artifact_metadata_by_type(job, artifact_type):
        return [artifact.metadata for artifact in job.artifacts if artifact.type == artifact_type]

    @classmethod
    def _get_checkpoints(cls, job):
        return cls._get_artifact_metadata_by_type(job, _ARTIFACT_TYPE_CHECKPOINT)

    @classmethod
    def _get_resources_allocated(cls, job):
        data = cls._get_artifact_metadata_by_type(job, _ARTIFACT_TYPE_RESOURCES_STATS)
        return data[0] if data else None

    @webmethod(route="/post-training/job/status")
    async def get_training_job_status(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]:
        job = self._scheduler.get_job(job_uuid)

        match job.status:
            # TODO: Add support for other statuses to API
            case SchedulerJobStatus.new | SchedulerJobStatus.scheduled:
                status = JobStatus.scheduled
            case SchedulerJobStatus.running:
                status = JobStatus.in_progress
            case SchedulerJobStatus.completed:
                status = JobStatus.completed
            case SchedulerJobStatus.failed:
                status = JobStatus.failed
            case _:
                raise NotImplementedError()

        return PostTrainingJobStatusResponse(
            job_uuid=job_uuid,
            status=status,
            scheduled_at=job.scheduled_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            checkpoints=self._get_checkpoints(job),
            resources_allocated=self._get_resources_allocated(job),
        )

    @webmethod(route="/post-training/job/cancel")
    async def cancel_training_job(self, job_uuid: str) -> None:
        self._scheduler.cancel(job_uuid)

    @webmethod(route="/post-training/job/artifacts")
    async def get_training_job_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]:
        job = self._scheduler.get_job(job_uuid)
        return PostTrainingJobArtifactsResponse(job_uuid=job_uuid, checkpoints=self._get_checkpoints(job))
