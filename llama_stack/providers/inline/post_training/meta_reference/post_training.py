# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.providers.inline.post_training.meta_reference.config import (
    MetaReferencePostTrainingConfig,
)
from llama_stack.apis.post_training import *  # noqa
from llama_stack.providers.inline.post_training.meta_reference.recipes.lora_finetuning_single_device import (
    LoraFinetuningSingleDevice,
)


class MetaReferencePostTrainingImpl:
    def __init__(
        self, config: MetaReferencePostTrainingConfig, datasetio_api: DatasetIO
    ) -> None:
        self.config = config
        self.datasetio_api = datasetio_api

    def supervised_fine_tune(
        self,
        job_uuid: str,
        model: str,
        dataset_id: str,
        validation_dataset_id: str,
        algorithm: FinetuningAlgorithm,
        algorithm_config: LoraFinetuningConfig,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob:
        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request = PostTrainingSFTRequest(
            job_uuid=job_uuid,
            model=model,
            dataset_id=dataset_id,
            validation_dataset_id=validation_dataset_id,
            algorithm=algorithm,
            algorithm_config=algorithm_config,
            optimizer_config=optimizer_config,
            training_config=training_config,
            logger_config=logger_config,
        )
        if request.algorithm == FinetuningAlgorithm.lora:
            recipe = LoraFinetuningSingleDevice(
                self.config, request, self.datasetio_api
            )
            recipe.setup(self.config)
            recipe.train()
        else:
            raise NotImplementedError()

        return PostTrainingJob(job_uuid=job_uuid)
