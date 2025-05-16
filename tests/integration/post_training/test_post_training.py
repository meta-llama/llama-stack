# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import sys
import time
import uuid

import pytest

from llama_stack.apis.post_training import (
    DataConfig,
    LoraFinetuningConfig,
    TrainingConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def capture_output(capsys):
    """Fixture to capture and display output during test execution."""
    yield
    captured = capsys.readouterr()
    if captured.out:
        print("\nCaptured stdout:", captured.out)
    if captured.err:
        print("\nCaptured stderr:", captured.err)


# Force flush stdout to see prints immediately
sys.stdout.reconfigure(line_buffering=True)

# How to run this test:
#
# pytest llama_stack/providers/tests/post_training/test_post_training.py
#   -m "torchtune_post_training_huggingface_datasetio"
#   -v -s --tb=short --disable-warnings


class TestPostTraining:
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "purpose, source",
        [
            (
                "post-training/messages",
                {
                    "type": "uri",
                    "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
                },
            ),
        ],
    )
    @pytest.mark.timeout(360)  # 6 minutes timeout
    def test_supervised_fine_tune(self, llama_stack_client, purpose, source):
        logger.info("Starting supervised fine-tuning test")

        # register dataset to train
        dataset = llama_stack_client.datasets.register(
            purpose=purpose,
            source=source,
        )
        logger.info(f"Registered dataset with ID: {dataset.identifier}")

        algorithm_config = LoraFinetuningConfig(
            type="LoRA",
            lora_attn_modules=["q_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            rank=8,
            alpha=16,
        )

        data_config = DataConfig(
            dataset_id=dataset.identifier,
            batch_size=1,
            shuffle=False,
            data_format="instruct",
        )

        # setup training config with minimal settings
        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            max_steps_per_epoch=1,
            gradient_accumulation_steps=1,
        )

        job_uuid = f"test-job{uuid.uuid4()}"
        logger.info(f"Starting training job with UUID: {job_uuid}")

        # train with HF trl SFTTrainer as the default
        _ = llama_stack_client.post_training.supervised_fine_tune(
            job_uuid=job_uuid,
            model="ibm-granite/granite-3.3-2b-instruct",
            algorithm_config=algorithm_config,
            training_config=training_config,
            hyperparam_search_config={},
            logger_config={},
            checkpoint_dir=None,
        )

        while True:
            status = llama_stack_client.post_training.job.status(job_uuid=job_uuid)
            if not status:
                logger.error("Job not found")
                break

            logger.info(f"Current status: {status}")
            if status.status == "completed":
                break

            logger.info("Waiting for job to complete...")
            time.sleep(10)  # Increased sleep time to reduce polling frequency

        artifacts = llama_stack_client.post_training.job.artifacts(job_uuid=job_uuid)
        logger.info(f"Job artifacts: {artifacts}")

    # TODO: Fix these tests to properly represent the Jobs API in training
    # @pytest.mark.asyncio
    # async def test_get_training_jobs(self, post_training_stack):
    #     post_training_impl = post_training_stack
    #     jobs_list = await post_training_impl.get_training_jobs()
    #     assert isinstance(jobs_list, list)
    #     assert jobs_list[0].job_uuid == "1234"

    # @pytest.mark.asyncio
    # async def test_get_training_job_status(self, post_training_stack):
    #     post_training_impl = post_training_stack
    #     job_status = await post_training_impl.get_training_job_status("1234")
    #     assert isinstance(job_status, PostTrainingJobStatusResponse)
    #     assert job_status.job_uuid == "1234"
    #     assert job_status.status == JobStatus.completed
    #     assert isinstance(job_status.checkpoints[0], Checkpoint)

    # @pytest.mark.asyncio
    # async def test_get_training_job_artifacts(self, post_training_stack):
    #     post_training_impl = post_training_stack
    #     job_artifacts = await post_training_impl.get_training_job_artifacts("1234")
    #     assert isinstance(job_artifacts, PostTrainingJobArtifactsResponse)
    #     assert job_artifacts.job_uuid == "1234"
    #     assert isinstance(job_artifacts.checkpoints[0], Checkpoint)
    #     assert job_artifacts.checkpoints[0].identifier == "instructlab/granite-7b-lab"
    #     assert job_artifacts.checkpoints[0].epoch == 0
    # assert "/.llama/checkpoints/Llama3.2-3B-Instruct-sft-0" in job_artifacts.checkpoints[0].path
