# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import unittest
import warnings
from unittest.mock import patch

import pytest
from llama_stack_client.types.algorithm_config_param import LoraFinetuningConfig, QatFinetuningConfig
from llama_stack_client.types.post_training_supervised_fine_tune_params import (
    TrainingConfig,
    TrainingConfigDataConfig,
    TrainingConfigOptimizerConfig,
)

from llama_stack.providers.remote.post_training.nvidia.post_training import (
    ListNvidiaPostTrainingJobs,
    NvidiaPostTrainingAdapter,
    NvidiaPostTrainingConfig,
    NvidiaPostTrainingJob,
    NvidiaPostTrainingJobStatusResponse,
)


class TestNvidiaPostTraining(unittest.TestCase):
    def setUp(self):
        os.environ["NVIDIA_BASE_URL"] = "http://nemo.test"  # needed for llm inference
        os.environ["NVIDIA_CUSTOMIZER_URL"] = "http://nemo.test"  # needed for nemo customizer

        config = NvidiaPostTrainingConfig(
            base_url=os.environ["NVIDIA_BASE_URL"], customizer_url=os.environ["NVIDIA_CUSTOMIZER_URL"], api_key=None
        )
        self.adapter = NvidiaPostTrainingAdapter(config)
        self.make_request_patcher = patch(
            "llama_stack.providers.remote.post_training.nvidia.post_training.NvidiaPostTrainingAdapter._make_request"
        )
        self.mock_make_request = self.make_request_patcher.start()

    def tearDown(self):
        self.make_request_patcher.stop()

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, run_async):
        self.run_async = run_async

    def _assert_request(self, mock_call, expected_method, expected_path, expected_params=None, expected_json=None):
        """Helper method to verify request details in mock calls."""
        call_args = mock_call.call_args

        if expected_method and expected_path:
            if isinstance(call_args[0], tuple) and len(call_args[0]) == 2:
                assert call_args[0] == (expected_method, expected_path)
            else:
                assert call_args[1]["method"] == expected_method
                assert call_args[1]["path"] == expected_path

        if expected_params:
            assert call_args[1]["params"] == expected_params

        if expected_json:
            for key, value in expected_json.items():
                assert call_args[1]["json"][key] == value

    def test_supervised_fine_tune(self):
        """Test the supervised fine-tuning API call."""
        self.mock_make_request.return_value = {
            "id": "cust-JGTaMbJMdqjJU8WbQdN9Q2",
            "created_at": "2024-12-09T04:06:28.542884",
            "updated_at": "2024-12-09T04:06:28.542884",
            "config": {
                "schema_version": "1.0",
                "id": "af783f5b-d985-4e5b-bbb7-f9eec39cc0b1",
                "created_at": "2024-12-09T04:06:28.542657",
                "updated_at": "2024-12-09T04:06:28.569837",
                "custom_fields": {},
                "name": "meta-llama/Llama-3.1-8B-Instruct",
                "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                "model_path": "llama-3_1-8b-instruct",
                "training_types": [],
                "finetuning_types": ["lora"],
                "precision": "bf16",
                "num_gpus": 4,
                "num_nodes": 1,
                "micro_batch_size": 1,
                "tensor_parallel_size": 1,
                "max_seq_length": 4096,
            },
            "dataset": {
                "schema_version": "1.0",
                "id": "dataset-XU4pvGzr5tvawnbVxeJMTb",
                "created_at": "2024-12-09T04:06:28.542657",
                "updated_at": "2024-12-09T04:06:28.542660",
                "custom_fields": {},
                "name": "sample-basic-test",
                "version_id": "main",
                "version_tags": [],
            },
            "hyperparameters": {
                "finetuning_type": "lora",
                "training_type": "sft",
                "batch_size": 16,
                "epochs": 2,
                "learning_rate": 0.0001,
                "lora": {"adapter_dim": 16, "adapter_dropout": 0.1},
            },
            "output_model": "default/job-1234",
            "status": "created",
            "project": "default",
            "custom_fields": {},
            "ownership": {"created_by": "me", "access_policies": {}},
        }

        algorithm_config = LoraFinetuningConfig(
            type="LoRA",
            adapter_dim=16,
            adapter_dropout=0.1,
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            alpha=16,
            rank=16,
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        data_config = TrainingConfigDataConfig(dataset_id="sample-basic-test", batch_size=16)

        optimizer_config = TrainingConfigOptimizerConfig(
            lr=0.0001,
        )

        training_config = TrainingConfig(
            n_epochs=2,
            data_config=data_config,
            optimizer_config=optimizer_config,
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            training_job = self.run_async(
                self.adapter.supervised_fine_tune(
                    job_uuid="1234",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    checkpoint_dir="",
                    algorithm_config=algorithm_config,
                    training_config=training_config,
                    logger_config={},
                    hyperparam_search_config={},
                )
            )

        # check the output is a PostTrainingJob
        assert isinstance(training_job, NvidiaPostTrainingJob)
        assert training_job.job_uuid == "cust-JGTaMbJMdqjJU8WbQdN9Q2"

        self.mock_make_request.assert_called_once()
        self._assert_request(
            self.mock_make_request,
            "POST",
            "/v1/customization/jobs",
            expected_json={
                "config": "meta/llama-3.1-8b-instruct",
                "dataset": {"name": "sample-basic-test", "namespace": "default"},
                "hyperparameters": {
                    "training_type": "sft",
                    "finetuning_type": "lora",
                    "epochs": 2,
                    "batch_size": 16,
                    "learning_rate": 0.0001,
                    "lora": {"alpha": 16, "adapter_dim": 16, "adapter_dropout": 0.1},
                },
            },
        )

    def test_supervised_fine_tune_with_qat(self):
        algorithm_config = QatFinetuningConfig(type="QAT", quantizer_name="quantizer_name", group_size=1)
        data_config = TrainingConfigDataConfig(dataset_id="sample-basic-test", batch_size=16)
        optimizer_config = TrainingConfigOptimizerConfig(
            lr=0.0001,
        )
        training_config = TrainingConfig(
            n_epochs=2,
            data_config=data_config,
            optimizer_config=optimizer_config,
        )
        # This will raise NotImplementedError since QAT is not supported
        with self.assertRaises(NotImplementedError):
            self.run_async(
                self.adapter.supervised_fine_tune(
                    job_uuid="1234",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    checkpoint_dir="",
                    algorithm_config=algorithm_config,
                    training_config=training_config,
                    logger_config={},
                    hyperparam_search_config={},
                )
            )

    def test_get_training_job_status(self):
        self.mock_make_request.return_value = {
            "created_at": "2024-12-09T04:06:28.580220",
            "updated_at": "2024-12-09T04:21:19.852832",
            "status": "completed",
            "steps_completed": 1210,
            "epochs_completed": 2,
            "percentage_done": 100.0,
            "best_epoch": 2,
            "train_loss": 1.718016266822815,
            "val_loss": 1.8661999702453613,
        }

        job_id = "cust-JGTaMbJMdqjJU8WbQdN9Q2"

        status = self.run_async(self.adapter.get_training_job_status(job_uuid=job_id))

        assert isinstance(status, NvidiaPostTrainingJobStatusResponse)
        assert status.status.value == "completed"
        assert status.steps_completed == 1210
        assert status.epochs_completed == 2
        assert status.percentage_done == 100.0
        assert status.best_epoch == 2
        assert status.train_loss == 1.718016266822815
        assert status.val_loss == 1.8661999702453613

        self.mock_make_request.assert_called_once()
        self._assert_request(
            self.mock_make_request, "GET", f"/v1/customization/jobs/{job_id}/status", expected_params={"job_id": job_id}
        )

    def test_get_training_jobs(self):
        job_id = "cust-JGTaMbJMdqjJU8WbQdN9Q2"
        self.mock_make_request.return_value = {
            "data": [
                {
                    "id": job_id,
                    "created_at": "2024-12-09T04:06:28.542884",
                    "updated_at": "2024-12-09T04:21:19.852832",
                    "config": {
                        "name": "meta-llama/Llama-3.1-8B-Instruct",
                        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
                    },
                    "dataset": {"name": "default/sample-basic-test"},
                    "hyperparameters": {
                        "finetuning_type": "lora",
                        "training_type": "sft",
                        "batch_size": 16,
                        "epochs": 2,
                        "learning_rate": 0.0001,
                        "lora": {"adapter_dim": 16, "adapter_dropout": 0.1},
                    },
                    "output_model": "default/job-1234",
                    "status": "completed",
                    "project": "default",
                }
            ]
        }

        jobs = self.run_async(self.adapter.get_training_jobs())

        assert isinstance(jobs, ListNvidiaPostTrainingJobs)
        assert len(jobs.data) == 1
        job = jobs.data[0]
        assert job.job_uuid == job_id
        assert job.status.value == "completed"

        self.mock_make_request.assert_called_once()
        self._assert_request(
            self.mock_make_request,
            "GET",
            "/v1/customization/jobs",
            expected_params={"page": 1, "page_size": 10, "sort": "created_at"},
        )

    def test_cancel_training_job(self):
        self.mock_make_request.return_value = {}  # Empty response for successful cancellation
        job_id = "cust-JGTaMbJMdqjJU8WbQdN9Q2"

        result = self.run_async(self.adapter.cancel_training_job(job_uuid=job_id))

        assert result is None

        self.mock_make_request.assert_called_once()
        self._assert_request(
            self.mock_make_request,
            "POST",
            f"/v1/customization/jobs/{job_id}/cancel",
            expected_params={"job_id": job_id},
        )


if __name__ == "__main__":
    unittest.main()
