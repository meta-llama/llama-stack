# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import base64
import mimetypes
import os

import pytest

from llama_stack.apis.common.job_types import JobStatus
from llama_stack.apis.post_training import (
    DataConfig,
    LoraFinetuningConfig,
    OptimizerConfig,
    TrainingConfig,
)

# How to run this test:
#
# pytest llama_stack/providers/tests/post_training/test_post_training.py
#   -m "torchtune_post_training_huggingface_datasetio"
#   -v -s --tb=short --disable-warnings


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


class TestPostTraining:
    @pytest.mark.asyncio
    def test_supervised_fine_tune(self, client_with_models):
        dataset = client_with_models.datasets.register(
            purpose="post-training/messages",
            source={
                "type": "uri",
                "uri": data_url_from_file(
                    os.path.join(os.path.dirname(__file__),
                                 "../datasets/test_dataset.csv")
                ),
            },
        )

        algorithm_config = LoraFinetuningConfig(
            type="LoRA",
            lora_attn_modules=["q_proj", "v_proj", "output_proj"],
            apply_lora_to_mlp=True,
            apply_lora_to_output=False,
            rank=1,
            alpha=1,
        )

        data_config = DataConfig(
            dataset_id=dataset.identifier,
            data_format="instruct",
            batch_size=1,
            shuffle=False,
        )

        optimizer_config = OptimizerConfig(
            optimizer_type="adamw",
            lr=3e-4,
            lr_min=3e-5,
            weight_decay=0.1,
            num_warmup_steps=1,
        )

        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            optimizer_config=optimizer_config,
            max_steps_per_epoch=1,
            max_validation_steps=1,
            gradient_accumulation_steps=1,
            dtype="fp32",
        )
        job = client_with_models.post_training.supervised_fine_tune(
            job_uuid="1234",
            model="Llama3.2-3B-Instruct",
            algorithm_config=algorithm_config,
            training_config=training_config,
            hyperparam_search_config={},
            logger_config={},
            checkpoint_dir="null",
        )
        assert job.job_uuid == "1234"

    @pytest.mark.asyncio
    def test_get_training_jobs(self, client_with_models):
        jobs_list = client_with_models.post_training.job.list()
        assert len(jobs_list) == 1
        assert jobs_list[0].job_uuid == "1234"

    @pytest.mark.asyncio
    def test_get_training_job_status(self, client_with_models):
        job_status = client_with_models.post_training.job.status(job_uuid="1234")
        assert job_status.job_uuid == "1234"
        assert job_status.status == JobStatus.completed.value

    @pytest.mark.asyncio
    def test_get_training_job_artifacts(self, client_with_models):
        job_artifacts = client_with_models.post_training.job.artifacts(job_uuid="1234")
        assert job_artifacts.job_uuid == "1234"
        assert job_artifacts.checkpoints[0]['identifier'] == "Llama3.2-3B-Instruct-sft-0"
        assert job_artifacts.checkpoints[0]['epoch'] == 0
        assert "/.llama/checkpoints/Llama3.2-3B-Instruct-sft-0" in job_artifacts.checkpoints[0]['path']
