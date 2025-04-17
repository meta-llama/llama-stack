# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import warnings
from typing import Any, Dict, Optional

import aiohttp

from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    TrainingConfig,
)
from llama_stack.providers.remote.post_training.nvidia.config import NvidiaPostTrainingConfig
from llama_stack.providers.remote.post_training.nvidia.utils import warn_unsupported_params
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper

from .models import _MODEL_ENTRIES


class NvidiaPostTrainingAdapter(ModelRegistryHelper):
    def __init__(self, config: NvidiaPostTrainingConfig):
        self.config = config
        self.headers = {}
        if config.api_key:
            self.headers["Authorization"] = f"Bearer {config.api_key}"

        self.timeout = aiohttp.ClientTimeout(total=config.timeout)
        # TODO: filter by available models based on /config endpoint
        ModelRegistryHelper.__init__(self, model_entries=_MODEL_ENTRIES)
        self.session = aiohttp.ClientSession(headers=self.headers, timeout=self.timeout)
        self.customizer_url = config.customizer_url

        if not self.customizer_url:
            warnings.warn("Customizer URL is not set, using default value: http://nemo.test", stacklevel=2)
            self.customizer_url = "http://nemo.test"

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
        url = f"{self.customizer_url}{path}"
        request_headers = self.headers.copy()

        if headers:
            request_headers.update(headers)

        # Add content-type header for JSON requests
        if json and "Content-Type" not in request_headers:
            request_headers["Content-Type"] = "application/json"

        for _ in range(self.config.max_retries):
            async with self.session.request(method, url, params=params, json=json, **kwargs) as response:
                if response.status >= 400:
                    error_data = await response.json()
                    raise Exception(f"API request failed: {error_data}")
                return await response.json()

        raise Exception(f"API request failed after {self.config.max_retries} retries")

    @staticmethod
    def _get_job_status(job: Dict[str, Any]) -> JobStatus:
        job_status = job.get("status", "unknown").lower()
        try:
            return JobStatus(job_status)
        except ValueError:
            return JobStatus.unknown

    # TODO: fetch just the necessary job from remote
    async def get_post_training_job(self, job_id: str) -> PostTrainingJob:
        jobs = await self.list_post_training_jobs()
        for job in jobs.data:
            if job.id == job_id:
                return job
        raise ValueError(f"Job with ID {job_id} not found")

    async def list_post_training_jobs(self) -> ListPostTrainingJobsResponse:
        """Get all customization jobs.

        ToDo: Support for schema input for filtering.
        """
        # TODO: don't hardcode pagination params
        params = {"page": 1, "page_size": 10, "sort": "created_at"}
        response = await self._make_request("GET", "/v1/customization/jobs", params=params)

        jobs = []
        for job_dict in response.get("data", []):
            # TODO: expose artifacts
            job = PostTrainingJob(**job_dict)
            job.update_status(self._get_job_status(job_dict))
            jobs.append(job)

        return ListPostTrainingJobsResponse(data=jobs)

    async def update_post_training_job(self, job_id: str, status: JobStatus | None = None) -> PostTrainingJob:
        if status is None:
            raise ValueError("Status must be provided")
        if status not in {JobStatus.cancelled}:
            raise ValueError(f"Unsupported status: {status}")
        await self._make_request(
            method="POST", path=f"/v1/customization/jobs/{job_id}/cancel", params={"job_id": job_id}
        )
        return await self.get_post_training_job(job_id)

    async def delete_post_training_job(self, job_id: str) -> None:
        raise NotImplementedError("Delete job is not implemented yet")

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: Dict[str, Any],
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        model: str,
        checkpoint_dir: Optional[str],
        algorithm_config: Optional[AlgorithmConfig] = None,
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
            checkpoint_dir: Optional[str] - Directory containing model checkpoints, ignored atm
            job_uuid: str - Unique identifier for the job, ignored atm
            hyperparam_search_config: Dict[str, Any] - Configuration for hyperparameter search, ignored atm
            logger_config: Dict[str, Any] - Configuration for logging, ignored atm

        Environment Variables:
            - NVIDIA_API_KEY: str - API key for the NVIDIA API
                Default: None
            - NVIDIA_DATASET_NAMESPACE: str - Namespace of the dataset
                Default: "default"
            - NVIDIA_CUSTOMIZER_URL: str - URL of the NeMo Customizer API
                Default: "http://nemo.test"
            - NVIDIA_PROJECT_ID: str - ID of the project
                Default: "test-project"
            - NVIDIA_OUTPUT_MODEL_DIR: str - Directory to save the output model
                Default: "test-example-model@v1"

        Supported models:
            - meta/llama-3.1-8b-instruct

        Supported algorithm configs:
            - LoRA, SFT

        Supported Parameters:
            - TrainingConfig:
                - n_epochs: int - Number of epochs to train
                    Default: 50
                - data_config: DataConfig - Configuration for the dataset
                - optimizer_config: OptimizerConfig - Configuration for the optimizer
                - dtype: str - Data type for training
                    not supported (users are informed via warnings)
                - efficiency_config: EfficiencyConfig - Configuration for efficiency
                    not supported
                - max_steps_per_epoch: int - Maximum number of steps per epoch
                    Default: 1000
                ## NeMo customizer specific parameters
                - log_every_n_steps: int - Log every n steps
                    Default: None
                - val_check_interval: float - Validation check interval
                    Default: 0.25
                - sequence_packing_enabled: bool - Sequence packing enabled
                    Default: False
                ## NeMo customizer specific SFT parameters
                - hidden_dropout: float - Hidden dropout
                    Default: None (0.0-1.0)
                - attention_dropout: float - Attention dropout
                    Default: None (0.0-1.0)
                - ffn_dropout: float - FFN dropout
                    Default: None (0.0-1.0)

            - DataConfig:
                - dataset_id: str - Dataset ID
                - batch_size: int - Batch size
                    Default: 8

            - OptimizerConfig:
                - lr: float - Learning rate
                    Default: 0.0001
                ## NeMo customizer specific parameter
                - weight_decay: float - Weight decay
                    Default: 0.01

            - LoRA config:
                ## NeMo customizer specific LoRA parameters
                - adapter_dim: int - Adapter dimension
                    Default: 8 (supports powers of 2)
                - adapter_dropout: float - Adapter dropout
                    Default: None (0.0-1.0)
                - alpha: int - Scaling factor for the LoRA update
                    Default: 16
            Note:
                - checkpoint_dir, hyperparam_search_config, logger_config are not supported (users are informed via warnings)
                - Some parameters from TrainingConfig, DataConfig, OptimizerConfig are not supported (users are informed via warnings)

            User is informed about unsupported parameters via warnings.
        """
        # Map model to nvidia model name
        # ToDo: only supports llama-3.1-8b-instruct now, need to update this to support other models
        nvidia_model = self.get_provider_model_id(model)

        # Check for unsupported method parameters
        unsupported_method_params = []
        if checkpoint_dir:
            unsupported_method_params.append(f"checkpoint_dir={checkpoint_dir}")
        if hyperparam_search_config:
            unsupported_method_params.append("hyperparam_search_config")
        if logger_config:
            unsupported_method_params.append("logger_config")

        if unsupported_method_params:
            warnings.warn(
                f"Parameters: {', '.join(unsupported_method_params)} are not supported and will be ignored",
                stacklevel=2,
            )

        # Define all supported parameters
        supported_params = {
            "training_config": {
                "n_epochs",
                "data_config",
                "optimizer_config",
                "log_every_n_steps",
                "val_check_interval",
                "sequence_packing_enabled",
                "hidden_dropout",
                "attention_dropout",
                "ffn_dropout",
            },
            "data_config": {"dataset_id", "batch_size"},
            "optimizer_config": {"lr", "weight_decay"},
            "lora_config": {"type", "adapter_dim", "adapter_dropout", "alpha"},
        }

        # Validate all parameters at once
        warn_unsupported_params(training_config, supported_params["training_config"], "TrainingConfig")
        warn_unsupported_params(training_config["data_config"], supported_params["data_config"], "DataConfig")
        warn_unsupported_params(
            training_config["optimizer_config"], supported_params["optimizer_config"], "OptimizerConfig"
        )

        output_model = self.config.output_model_dir

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
                **{
                    k: v
                    for k, v in {
                        "epochs": training_config.get("n_epochs"),
                        "batch_size": training_config["data_config"].get("batch_size"),
                        "learning_rate": training_config["optimizer_config"].get("lr"),
                        "weight_decay": training_config["optimizer_config"].get("weight_decay"),
                        "log_every_n_steps": training_config.get("log_every_n_steps"),
                        "val_check_interval": training_config.get("val_check_interval"),
                        "sequence_packing_enabled": training_config.get("sequence_packing_enabled"),
                    }.items()
                    if v is not None
                },
            },
            "project": self.config.project_id,
            # TODO: ignored ownership, add it later
            # "ownership": {"created_by": self.config.user_id, "access_policies": self.config.access_policies},
            "output_model": output_model,
        }

        # Handle SFT-specific optional parameters
        job_config["hyperparameters"]["sft"] = {
            k: v
            for k, v in {
                "ffn_dropout": training_config.get("ffn_dropout"),
                "hidden_dropout": training_config.get("hidden_dropout"),
                "attention_dropout": training_config.get("attention_dropout"),
            }.items()
            if v is not None
        }

        # Remove the sft dictionary if it's empty
        if not job_config["hyperparameters"]["sft"]:
            job_config["hyperparameters"].pop("sft")

        # Handle LoRA-specific configuration
        if algorithm_config:
            if isinstance(algorithm_config, dict) and algorithm_config.get("type") == "LoRA":
                warn_unsupported_params(algorithm_config, supported_params["lora_config"], "LoRA config")
                job_config["hyperparameters"]["lora"] = {
                    k: v
                    for k, v in {
                        "adapter_dim": algorithm_config.get("adapter_dim"),
                        "alpha": algorithm_config.get("alpha"),
                        "adapter_dropout": algorithm_config.get("adapter_dropout"),
                    }.items()
                    if v is not None
                }
            else:
                raise NotImplementedError(f"Unsupported algorithm config: {algorithm_config}")

        # Create the customization job
        response = await self._make_request(
            method="POST",
            path="/v1/customization/jobs",
            headers={"Accept": "application/json"},
            json=job_config,
        )
        response.pop("status")

        # TODO: expose artifacts
        job = PostTrainingJob(**response)
        job.update_status(JobStatus.running)
        return job

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
