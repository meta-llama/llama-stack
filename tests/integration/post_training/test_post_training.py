# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone

import pytest

from llama_stack.apis.common.job_types import JobStatus
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
    job_uuid = f"test-job{uuid.uuid4()}"
    model = "ibm-granite/granite-3.3-2b-instruct"

    def _validate_checkpoints(self, checkpoints):
        assert len(checkpoints) == 1
        assert checkpoints[0]["identifier"] == f"{self.model}-sft-1"
        assert checkpoints[0]["epoch"] == 1
        assert "/.llama/checkpoints/merged_model" in checkpoints[0]["path"]

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

        logger.info(f"Starting training job with UUID: {self.job_uuid}")

        # train with HF trl SFTTrainer as the default
        checkpoint_dir = os.path.expanduser("/mnt/")
        # os.makedirs(checkpoint_dir, exist_ok=True)

        started = datetime.now(timezone.utc)
        _ = llama_stack_client.post_training.supervised_fine_tune(
            job_uuid=self.job_uuid,
            model=self.model,
            algorithm_config=algorithm_config,
            training_config=training_config,
            hyperparam_search_config={},
            logger_config={},
            checkpoint_dir=checkpoint_dir,
        )

        while True:
            status = llama_stack_client.post_training.job.status(job_uuid=self.job_uuid)
            if not status:
                logger.error("Job not found")
                break

            logger.info(f"Current status: {status}")
            if status.status == "completed":
                completed = datetime.now(timezone.utc)
                assert status.completed_at is not None
                assert status.completed_at >= started
                assert status.completed_at <= completed
                break

            logger.info("Waiting for job to complete...")
            time.sleep(10)  # Increased sleep time to reduce polling frequency

    @pytest.mark.asyncio
    def test_get_training_jobs(self, client_with_models):
        jobs_list = client_with_models.post_training.job.list()
        assert len(jobs_list) == 1
        assert jobs_list[0].job_uuid == self.job_uuid

    @pytest.mark.asyncio
    def test_get_training_job_status(self, client_with_models):
        job_status = client_with_models.post_training.job.status(job_uuid=self.job_uuid)
        assert job_status.job_uuid == self.job_uuid
        assert job_status.status == JobStatus.completed.value
        assert isinstance(job_status.resources_allocated, dict)
        self._validate_checkpoints(job_status.checkpoints)

        assert job_status.scheduled_at is not None
        assert job_status.started_at is not None
        assert job_status.completed_at is not None

        assert job_status.scheduled_at <= job_status.started_at
        assert job_status.started_at <= job_status.completed_at

    @pytest.mark.asyncio
    def test_get_training_job_artifacts(self, client_with_models):
        job_artifacts = client_with_models.post_training.job.artifacts(job_uuid=self.job_uuid)
        assert job_artifacts.job_uuid == self.job_uuid
        self._validate_checkpoints(job_artifacts.checkpoints)
