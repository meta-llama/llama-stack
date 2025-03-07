# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import List

import pytest

from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.post_training import (
    Checkpoint,
    DataConfig,
    LoraFinetuningConfig,
    OptimizerConfig,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    TrainingConfig,
)

# How to run this test:
#
# pytest llama_stack/providers/tests/post_training/test_post_training.py
#   -m "torchtune_post_training_huggingface_datasetio"
#   -v -s --tb=short --disable-warnings


@pytest.mark.skip(reason="FIXME FIXME @yanxi0830 this needs to be migrated to use the API")
class TestPostTraining:
    @pytest.mark.asyncio
    async def test_supervised_fine_tune(self, post_training_stack):
        algorithm_config = LoraFinetuningConfig(
            type="LoRA",
            lora_attn_modules=["q_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            rank=8,
            alpha=16,
        )

        data_config = DataConfig(
            dataset_id="alpaca",
            batch_size=1,
            shuffle=False,
        )

        optimizer_config = OptimizerConfig(
            optimizer_type="adamw",
            lr=3e-4,
            lr_min=3e-5,
            weight_decay=0.1,
            num_warmup_steps=100,
        )

        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            optimizer_config=optimizer_config,
            max_steps_per_epoch=1,
            gradient_accumulation_steps=1,
        )
        post_training_impl = post_training_stack
        response = await post_training_impl.supervised_fine_tune(
            job_uuid="1234",
            model="Llama3.2-3B-Instruct",
            algorithm_config=algorithm_config,
            training_config=training_config,
            hyperparam_search_config={},
            logger_config={},
            checkpoint_dir="null",
        )
        assert isinstance(response, PostTrainingJob)
        assert response.job_uuid == "1234"

    @pytest.mark.asyncio
    async def test_get_training_jobs(self, post_training_stack):
        post_training_impl = post_training_stack
        jobs_list = await post_training_impl.get_training_jobs()
        assert isinstance(jobs_list, List)
        assert jobs_list[0].job_uuid == "1234"

    @pytest.mark.asyncio
    async def test_get_training_job_status(self, post_training_stack):
        post_training_impl = post_training_stack
        job_status = await post_training_impl.get_training_job_status("1234")
        assert isinstance(job_status, PostTrainingJobStatusResponse)
        assert job_status.job_uuid == "1234"
        assert job_status.status == JobStatus.completed
        assert isinstance(job_status.checkpoints[0], Checkpoint)

    @pytest.mark.asyncio
    async def test_get_training_job_artifacts(self, post_training_stack):
        post_training_impl = post_training_stack
        job_artifacts = await post_training_impl.get_training_job_artifacts("1234")
        assert isinstance(job_artifacts, PostTrainingJobArtifactsResponse)
        assert job_artifacts.job_uuid == "1234"
        assert isinstance(job_artifacts.checkpoints[0], Checkpoint)
        assert job_artifacts.checkpoints[0].identifier == "Llama3.2-3B-Instruct-sft-0"
        assert job_artifacts.checkpoints[0].epoch == 0
        assert "/.llama/checkpoints/Llama3.2-3B-Instruct-sft-0" in job_artifacts.checkpoints[0].path
