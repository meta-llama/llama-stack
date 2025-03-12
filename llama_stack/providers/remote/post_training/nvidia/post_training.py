# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import warnings
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import aiohttp
from pydantic import BaseModel, ConfigDict

from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    JobStatus,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from llama_stack.providers.remote.post_training.nvidia.config import (
    NvidiaPostTrainingConfig,
)
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.schema_utils import webmethod

from .models import _MODEL_ENTRIES

# Map API status to JobStatus enum
STATUS_MAPPING = {
    "running": "in_progress",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "pending": "scheduled",
}


class NvidiaPostTrainingJob(PostTrainingJob):
    """Parse the response from the Customizer API.
    Inherits job_uuid from PostTrainingJob.
    Adds status, created_at, updated_at parameters.
    Passes through all other parameters from data field in the response.
    """

    model_config = ConfigDict(extra="allow")
    status: JobStatus
    created_at: datetime
    updated_at: datetime


class ListNvidiaPostTrainingJobs(BaseModel):
    data: List[NvidiaPostTrainingJob]


class NvidiaPostTrainingAdapter(ModelRegistryHelper):
    def __init__(self, config: NvidiaPostTrainingConfig):
        self.config = config
        self.headers = {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

        self.timeout = aiohttp.ClientTimeout(total=config.timeout)
        # TODO(mf): filter by available models
        ModelRegistryHelper.__init__(self, model_entries=_MODEL_ENTRIES)

    async def _make_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Helper method to make HTTP requests to the Customizer API."""
        url = f"{self.config.customizer_url}{path}"
        request_headers = self.headers.copy()  # Create a copy to avoid modifying the original

        if headers:
            request_headers.update(headers)

        # Add content-type header for JSON requests
        if json and "Content-Type" not in request_headers:
            request_headers["Content-Type"] = "application/json"

        for _ in range(self.config.max_retries):
            async with aiohttp.ClientSession(headers=request_headers, timeout=self.timeout) as session:
                async with session.request(method, url, params=params, json=json, **kwargs) as response:
                    if response.status >= 400:
                        error_data = await response.json()
                        raise Exception(f"API request failed: {error_data}")
                    return await response.json()

    @webmethod(route="/post-training/jobs", method="GET")
    async def get_training_jobs(
        self,
        page: Optional[int] = 1,
        page_size: Optional[int] = 10,
        sort: Optional[Literal["created_at", "-created_at"]] = "created_at",
    ) -> ListNvidiaPostTrainingJobs:
        """Get all customization jobs.
        Updated the base class return type from ListPostTrainingJobsResponse to ListNvidiaPostTrainingJobs.
        """
        params = {"page": page, "page_size": page_size, "sort": sort}

        response = await self._make_request("GET", "/v1/customization/jobs", params=params)

        jobs = []
        for job in response.get("data", []):
            job_id = job.pop("id")
            job_status = job.pop("status", "unknown").lower()
            mapped_status = STATUS_MAPPING.get(job_status, "unknown")

            # Convert string timestamps to datetime objects
            created_at = datetime.fromisoformat(job.pop("created_at")) if "created_at" in job else datetime.now()
            updated_at = datetime.fromisoformat(job.pop("updated_at")) if "updated_at" in job else datetime.now()

            # Create NvidiaPostTrainingJob instance
            jobs.append(
                NvidiaPostTrainingJob(
                    job_uuid=job_id,
                    status=JobStatus(mapped_status),
                    created_at=created_at,
                    updated_at=updated_at,
                    **job,
                )
            )

        return ListNvidiaPostTrainingJobs(data=jobs)

    @webmethod(route="/post-training/job/status", method="GET")
    async def get_training_job_status(self, job_uuid: str) -> Optional[NvidiaPostTrainingJob]:
        """Get the status of a customization job.
        Updated the base class return type from PostTrainingJobResponse to NvidiaPostTrainingJob.
        """
        response = await self._make_request(
            "GET",
            f"/v1/customization/jobs/{job_uuid}/status",
            params={"job_id": job_uuid},
        )

        api_status = response.pop("status").lower()
        mapped_status = STATUS_MAPPING.get(api_status, "unknown")

        return NvidiaPostTrainingJob(
            status=JobStatus(mapped_status),
            job_uuid=job_uuid,
            created_at=datetime.fromisoformat(response.pop("created_at")),
            updated_at=datetime.fromisoformat(response.pop("updated_at")),
            **response,
        )

    @webmethod(route="/post-training/job/cancel", method="POST")
    async def cancel_training_job(self, job_uuid: str) -> None:
        """Cancels a customization job."""
        await self._make_request(
            method="POST", path=f"/v1/customization/jobs/{job_uuid}/cancel", params={"job_id": job_uuid}
        )

    @webmethod(route="/post-training/job/artifacts")
    async def get_training_job_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]:
        """Get artifacts for a specific training job."""
        raise NotImplementedError("Job artifacts are not implemented yet")

    @webmethod(route="/post-training/artifacts", method="GET")
    async def get_post_training_artifacts(self, job_uuid: str) -> Optional[PostTrainingJobArtifactsResponse]:
        """Get all post-training artifacts."""
        raise NotImplementedError("Job artifacts are not implemented yet")

    @webmethod(route="/post-training/supervised-fine-tune", method="POST")
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
            - nemo microservice is running and endpoint is set in config.customizer_url
            - dataset is registered separately in nemo datastore
            - model checkpoint is downloaded as per nemo customizer requirements

        Parameters:
            training_config: TrainingConfig - Configuration for training
            model: str - Model identifier
            algorithm_config: Optional[AlgorithmConfig] - Algorithm-specific configuration
            checkpoint_dir: Optional[str] - Directory containing model checkpoints
            job_uuid: str - Unique identifier for the job
            hyperparam_search_config: Dict[str, Any] - Configuration for hyperparameter search
            logger_config: Dict[str, Any] - Configuration for logging

        Environment Variables:
            - NVIDIA_PROJECT_ID: ID of the project
            - NVIDIA_USER_ID: ID of the user
            - NVIDIA_ACCESS_POLICIES: Access policies for the project
            - NVIDIA_DATASET_NAMESPACE: Namespace of the dataset
            - NVIDIA_OUTPUT_MODEL_DIR: Directory to save the output model

        Supported models:
            - meta/llama-3.1-8b-instruct

        Supported algorithm configs:
            - LoRA, SFT

        Supported Parameters:
            - TrainingConfig:
                - n_epochs
                - data_config
                - optimizer_config
                - dtype
                - efficiency_config
                - max_steps_per_epoch
            - DataConfig:
                - dataset_id
                - batch_size
            - OptimizerConfig:
                - lr
            - LoRA config:
                - adapter_dim
                - adapter_dropout
            Note:
                - checkpoint_dir, hyperparam_search_config, logger_config are not supported atm, will be ignored
                - output_model_dir is set via environment variable NVIDIA_OUTPUT_MODEL_DIR

            User is informed about unsupported parameters via warnings.
        """
        # map model to nvidia model name
        nvidia_model = self.get_provider_model_id(model)

        # Check for unsupported parameters
        if checkpoint_dir or hyperparam_search_config or logger_config:
            warnings.warn(
                "Parameters: {} not supported atm, will be ignored".format(
                    checkpoint_dir,
                )
            )

        def warn_unsupported_params(config_dict: Dict[str, Any], supported_keys: List[str], config_name: str) -> None:
            """Helper function to warn about unsupported parameters in a config dictionary."""
            unsupported_params = [k for k in config_dict.keys() if k not in supported_keys]
            if unsupported_params:
                warnings.warn(f"Parameters: {unsupported_params} in {config_name} not supported and will be ignored.")

        # Check for unsupported parameters
        warn_unsupported_params(training_config, ["n_epochs", "data_config", "optimizer_config"], "TrainingConfig")
        warn_unsupported_params(training_config["data_config"], ["dataset_id", "batch_size"], "DataConfig")
        warn_unsupported_params(training_config["optimizer_config"], ["lr"], "OptimizerConfig")

        output_model = self.config.output_model_dir

        if output_model == "default":
            warnings.warn("output_model_dir set via default value, will be ignored")

        # Prepare base job configuration
        job_config = {
            "config": nvidia_model,
            "dataset": {
                "name": training_config["data_config"]["dataset_id"],
                "namespace": self.config.dataset_namespace,
            },
            "hyperparameters": {
                "training_type": "sft",
                "finetuning_type": "lora",
                "epochs": training_config["n_epochs"],
                "batch_size": training_config["data_config"]["batch_size"],
                "learning_rate": training_config["optimizer_config"]["lr"],
            },
            "project": self.config.project_id,
            "ownership": {"created_by": self.config.user_id, "access_policies": self.config.access_policies},
            "output_model": output_model,
        }

        # Handle LoRA-specific configuration
        if algorithm_config:
            if isinstance(algorithm_config, dict) and algorithm_config.get("type") == "LoRA":
                # Extract LoRA-specific parameters
                lora_config = {k: v for k, v in algorithm_config.items() if k != "type"}
                job_config["hyperparameters"]["lora"] = lora_config
                warn_unsupported_params(lora_config, ["adapter_dim", "adapter_dropout"], "LoRA config")
            else:
                raise NotImplementedError(f"Unsupported algorithm config: {algorithm_config}")

        # Create the customization job
        response = await self._make_request(
            method="POST",
            path="/v1/customization/jobs",
            headers={"Accept": "application/json"},
            json=job_config,
        )

        job_uuid = response["id"]
        return PostTrainingJob(job_uuid=job_uuid)

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob:
        """Optimize a model based on preference data."""
        raise NotImplementedError("Preference optimization is not implemented yet")

    @webmethod(route="/post-training/job/logs", method="GET")
    async def get_training_job_container_logs(self, job_uuid: str) -> Optional[PostTrainingJobStatusResponse]:
        """Get the container logs of a customization job."""
        raise NotImplementedError("Job logs are not implemented yet")
