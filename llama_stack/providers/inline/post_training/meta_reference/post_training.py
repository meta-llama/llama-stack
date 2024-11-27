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

    LoraFinetuningConfig(
        lora_attn_modules=["q_proj", "v_proj", "output_proj"],
        apply_lora_to_mlp=True,
        apply_lora_to_output=False,
        rank=8,
        alpha=16,
    )

    OptimizerConfig(
        lr=3e-4,
        lr_min=3e-5,
        weight_decay=0.1,
        num_warmup_steps=100,
    )

    TrainingConfig(
        dtype="bf16",
        n_epochs=1,
        max_steps_per_epoch=10,
        gradient_accumulation_steps=1,
        batch_size=1,
        shuffle=1,
        enable_activation_checkpointing=False,
        memory_efficient_fsdp_wrap=False,
        fsdp_cpu_offload=False,
    )

    def supervised_fine_tune(
        self,
        job_uuid: str = "1234",
        model: str = " meta-llama/Llama-3.2-3B-Instruct",
        dataset_id: str = "alpaca",
        validation_dataset_id: str = "alpaca",
        algorithm: FinetuningAlgorithm = FinetuningAlgorithm.lora,
        algorithm_config: LoraFinetuningConfig = LoraFinetuningConfig,
        optimizer_config: OptimizerConfig = OptimizerConfig,
        training_config: TrainingConfig = TrainingConfig,
        hyperparam_search_config: Dict[str, Any] = {},
        logger_config: Dict[str, Any] = {},
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

    def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: URL,
        dataset_id: str,
        validation_dataset_id: str,
        algorithm: RLHFAlgorithm,
        algorithm_config: DPOAlignmentConfig,
        optimizer_config: OptimizerConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob: ...

    def get_training_jobs(self) -> List[PostTrainingJob]: ...

    # sends SSE stream of logs
    @webmethod(route="/post-training/job/logs")
    def get_training_job_logstream(self, job_uuid: str) -> PostTrainingJobLogStream: ...

    @webmethod(route="/post-training/job/status")
    def get_training_job_status(
        self, job_uuid: str
    ) -> PostTrainingJobStatusResponse: ...

    @webmethod(route="/post-training/job/cancel")
    def cancel_training_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/post-training/job/artifacts")
    def get_training_job_artifacts(
        self, job_uuid: str
    ) -> PostTrainingJobArtifactsResponse: ...
