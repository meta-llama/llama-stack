# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from datetime import datetime
from typing import Any, Dict, Literal, Optional

import aiohttp
from aiohttp import ClientTimeout

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
from llama_stack.providers.remote.post_training.nvidia.config import (
    NvidiaPostTrainingConfig,
)
from llama_stack.schema_utils import webmethod


class NvidiaPostTrainingImpl:
    def __init__(self, config: NvidiaPostTrainingConfig):
        self.config = config
        self.headers = {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

        self.timeout = ClientTimeout(total=config.timeout)

    async def _make_request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Helper method to make HTTP requests to the Customizer API."""
        url = f"{self.config.customizer_url}{path}"

        for attempt in range(self.config.max_retries):
            async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                async with session.request(method, url, **kwargs) as response:
                    if response.status >= 400:
                        error_data = response
                        raise Exception(f"API request failed: {error_data}")
                    return await response.json()

    @webmethod(route="/post-training/jobs", method="GET")
    async def get_training_jobs(
        self,
        page: int = 1,
        page_size: int = 10,
        sort: Literal[
            "created_at",
            "-created_at",
        ] = "created_at",
    ) -> ListPostTrainingJobsResponse:
        """
        Get all customization jobs.
        """
        params = {"page": page, "page_size": page_size, "sort": sort}

        response = await self._make_request(
            "GET",
            "/v1/customization/jobs",
            # params=params
        )

        # Convert customization jobs to PostTrainingJob objects
        jobs = [PostTrainingJob(job_uuid=job["id"]) for job in response["data"]]

        # Remove the data and pass through other fields
        response.pop("data")
        return ListPostTrainingJobsResponse(data=jobs, **response)

    @webmethod(route="/post-training/job/status", method="GET")
    async def get_training_job_status(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]:
        """
        Get the status of a customization job.
        """
        response = await self._make_request(
            "GET",
            f"/v1/customization/jobs/{job_uuid}/status",
            params=job_uuid,
        )

        # Map API status to JobStatus enum
        status_mapping = {
            "running": "in_progress",
            "completed": "completed",
            "failed": "failed",
            # "cancelled": "cancelled",
            "pending": "scheduled",
        }

        api_status = response["status"].lower()
        mapped_status = status_mapping.get(api_status, "unknown")

        # todo: add callback for rest of the parameters
        response["status"] = JobStatus(mapped_status)
        response["job_uuid"] = job_uuid
        response["started_at"] = datetime.fromisoformat(response["created_at"])

        return PostTrainingJobStatusResponse(**response)

    @webmethod(route="/post-training/job/cancel", method="POST")
    async def cancel_training_job(self, job_uuid: str) -> None:
        """
        Cancels a customization job.
        """
        response = await self._make_request(
            "POST", f"/v1/customization/jobs/{job_uuid}/cancel", params={"job_id": job_uuid}
        )

    @webmethod(route="/post-training/job/artifacts")
    async def get_training_job_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]:
        raise NotImplementedError("Job artifacts are not implemented yet")

    ## post-training artifacts
    @webmethod(route="/post-training/artifacts", method="GET")
    async def get_post_training_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]:
        raise NotImplementedError("Job artifacts are not implemented yet")

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
        """
        Fine-tunes a model on a dataset.
        Currently only supports Lora finetuning for standlone docker container.
        Assumptions:
            - model is a valid Nvidia model
            - dataset is registered separately in nemo datastore
            - model checkpoint is downloaded from ngc and exists in the local directory

            Parameters:
                training_config: TrainingConfig
                model: str
                algorithm_config: Optional[AlgorithmConfig]
                dataset: Optional[str]
                run_local_jobs: bool = False True for standalone mode.
                nemo_data_store_url: str URL of NeMo Data Store for Customizer to connect to for dataset and model files.

            LoRA config:
                training_type = sft
                finetuning_type = lora
                adapter_dim = Size of adapter layers added throughout the model.
                adapter_dropout = Dropout probability in the adapter layer.

            ToDo:
                support for model config of helm chart ??
                /status endpoint for model customization
                Get Metrics for customization
                Weights and Biases integration ??
                OpenTelemetry integration ??
        """
        # map model to nvidia model name
        model_mapping = {
            "Llama3.1-8B-Instruct": "meta/llama-3.1-8b-instruct",
        }
        nvidia_model = model_mapping.get(model, model)

        # Prepare the customization job request
        job_config = {
            "config": nvidia_model,
            "dataset": {
                "name": training_config["data_config"]["dataset_id"],
                "namespace": "default",  # todo: could be configurable in the future
            },
            "hyperparameters": {
                "training_type": "sft",
                "finetuning_type": "lora",
                "epochs": training_config["n_epochs"],
                "batch_size": training_config["data_config"]["batch_size"],
                "learning_rate": training_config["optimizer_config"]["lr"],
                "lora": {"adapter_dim": 16},
            },
            "project": "llama-stack-project",  # todo: could be configurable
            "ownership": {
                "created_by": self.config.user_id or "llama-stack-user",
            },
            "output_model": f"llama-stack-{training_config['data_config']['dataset_id']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        }

        # Add LoRA specific configuration if provided
        if isinstance(algorithm_config, Dict) and algorithm_config["type"] == "LoRA":
            if algorithm_config["adapter_dim"]:
                job_config["hyperparameters"]["lora"]["adapter_dim"] = algorithm_config["adapter_dim"]
        else:
            raise NotImplementedError(f"Algorithm config {type(algorithm_config)} not implemented.")

        # Make the API request to create the customization job
        response = await self._make_request("POST", "/v1/customization/jobs", json=job_config)

        # Parse the response to extract relevant fields
        job_uuid = response["id"]
        created_at = response["created_at"]
        status = response["status"]
        output_model = response["output_model"]
        project = response["project"]
        created_by = response["ownership"]["created_by"]

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

    @webmethod(route="/post-training/job/logs", method="GET")
    async def get_training_job_container_logs(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]:
        """Get the container logs of a customization job."""
        raise NotImplementedError("Job logs are not implemented yet")
