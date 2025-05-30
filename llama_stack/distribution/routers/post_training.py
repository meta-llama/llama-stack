# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.models import Model
from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    ListPostTrainingJobsResponse,
    PostTraining,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import RoutingTable

logger = get_logger(name=__name__, category="core")


class PostTrainingRouter(PostTraining):
    """Routes to an provider based on the model"""

    async def initialize(self) -> None:
        pass

    def __init__(
        self,
        routing_table: RoutingTable,
    ) -> None:
        logger.debug("Initializing InferenceRouter")
        self.routing_table = routing_table

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
        model: str,
        checkpoint_dir: str | None = None,
        algorithm_config: AlgorithmConfig | None = None,
    ) -> PostTrainingJob:
        provider = self.routing_table.get_provider_impl(model)
        params = dict(
            job_uuid=job_uuid,
            training_config=training_config,
            hyperparam_search_config=hyperparam_search_config,
            logger_config=logger_config,
            model=model,
            checkpoint_dir=checkpoint_dir,
            algorithm_config=algorithm_config,
        )
        return provider.supervised_fine_tune(**params)

    async def register_model(self, model: Model) -> Model:
        try:
            # get static list of models
            model = await self.register_helper.register_model(model)
        except ValueError:
            # if model is NOT in the list, its probably ok, but warn the user.
            #
            logger.warning(
                f"Model {model.identifier} is not in the model registry for this provider, there might be unexpected issues."
            )
        if model.provider_resource_id is None:
            raise ValueError("Model provider_resource_id cannot be None")
        provider_resource_id = self.register_helper.get_provider_model_id(model.provider_resource_id)
        if provider_resource_id is None:
            provider_resource_id = model.provider_resource_id
        model.provider_resource_id = provider_resource_id

        return model

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: dict[str, Any],
        logger_config: dict[str, Any],
    ) -> PostTrainingJob:
        pass

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        pass

    async def get_training_job_status(self, job_uuid: str) -> PostTrainingJobStatusResponse | None:
        pass

    async def cancel_training_job(self, job_uuid: str) -> None:
        pass

    async def get_training_job_artifacts(self, job_uuid: str) -> PostTrainingJobArtifactsResponse | None:
        pass
