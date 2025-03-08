# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_stack_client.types.algorithm_config_param import LoraFinetuningConfig
from llama_stack_client.types.post_training_supervised_fine_tune_params import (
    TrainingConfig,
    TrainingConfigDataConfig,
    TrainingConfigOptimizerConfig,
)

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

POST_TRAINING_PROVIDER_TYPES = ["remote::nvidia"]


@pytest.mark.integration
@pytest.fixture(scope="session")
def post_training_provider_available(llama_stack_client):
    providers = llama_stack_client.providers.list()
    post_training_providers = [p for p in providers if p.provider_type in POST_TRAINING_PROVIDER_TYPES]
    return len(post_training_providers) > 0


@pytest.mark.integration
def test_post_training_provider_registration(llama_stack_client, post_training_provider_available):
    """Check if post_training is in the api list.
    This is a sanity check to ensure the provider is registered."""
    if not post_training_provider_available:
        pytest.skip("post training provider not available")

    providers = llama_stack_client.providers.list()
    post_training_providers = [p for p in providers if p.provider_type in POST_TRAINING_PROVIDER_TYPES]
    assert len(post_training_providers) > 0


class TestNvidiaPostTraining(unittest.TestCase):
    def setUp(self):
        os.environ["NVIDIA_CUSTOMIZER_URL"] = "http://nemo.test"
        os.environ["NVIDIA_BASE_URL"] = "http://nim.test"

        self.llama_stack_client = LlamaStackAsLibraryClient("nvidia")

        self.llama_stack_client.initialize = MagicMock(return_value=None)
        _ = self.llama_stack_client.initialize()

    @patch("requests.post")
    def test_supervised_fine_tune(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
                "name": "default/sample-basic-test",
                "version_id": "main",
                "version_tags": [],
            },
            "hyperparameters": {
                "finetuning_type": "lora",
                "training_type": "sft",
                "batch_size": 16,
                "epochs": 2,
                "learning_rate": 0.0001,
                "lora": {"adapter_dim": 16},
            },
            "output_model": "default/job-1234",
            "status": "created",
            "project": "default",
            "custom_fields": {},
            "ownership": {"created_by": "me", "access_policies": {}},
        }
        mock_post.return_value = mock_response

        algorithm_config = LoraFinetuningConfig(type="LoRA", adapter_dim=16)

        data_config = TrainingConfigDataConfig(dataset_id="sample-basic-test", batch_size=16)

        optimizer_config = TrainingConfigOptimizerConfig(
            lr=0.0001,
        )

        training_config = TrainingConfig(
            n_epochs=2,
            data_config=data_config,
            optimizer_config=optimizer_config,
        )

        with patch.object(
            self.llama_stack_client.post_training,
            "supervised_fine_tune",
            return_value={
                "id": "cust-JGTaMbJMdqjJU8WbQdN9Q2",
                "status": "created",
                "created_at": "2024-12-09T04:06:28.542884",
                "updated_at": "2024-12-09T04:06:28.542884",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "dataset_id": "sample-basic-test",
                "output_model": "default/job-1234",
            },
        ):
            training_job = self.llama_stack_client.post_training.supervised_fine_tune(
                job_uuid="1234",
                model="meta-llama/Llama-3.1-8B-Instruct",
                checkpoint_dir="",
                algorithm_config=algorithm_config,
                training_config=training_config,
                logger_config={},
                hyperparam_search_config={},
            )

            self.assertEqual(training_job["id"], "cust-JGTaMbJMdqjJU8WbQdN9Q2")
            self.assertEqual(training_job["status"], "created")
            self.assertEqual(training_job["model"], "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(training_job["dataset_id"], "sample-basic-test")

    @patch("requests.get")
    def test_get_job_status(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
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
        mock_get.return_value = mock_response

        with patch.object(
            self.llama_stack_client.post_training.job,
            "status",
            return_value={
                "id": "cust-JGTaMbJMdqjJU8WbQdN9Q2",
                "status": "completed",
                "created_at": "2024-12-09T04:06:28.580220",
                "updated_at": "2024-12-09T04:21:19.852832",
                "steps_completed": 1210,
                "epochs_completed": 2,
                "percentage_done": 100.0,
                "best_epoch": 2,
                "train_loss": 1.718016266822815,
                "val_loss": 1.8661999702453613,
            },
        ):
            status = self.llama_stack_client.post_training.job.status("cust-JGTaMbJMdqjJU8WbQdN9Q2")

            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["steps_completed"], 1210)
            self.assertEqual(status["epochs_completed"], 2)
            self.assertEqual(status["percentage_done"], 100.0)
            self.assertEqual(status["best_epoch"], 2)
            self.assertEqual(status["train_loss"], 1.718016266822815)
            self.assertEqual(status["val_loss"], 1.8661999702453613)

    @patch("requests.get")
    def test_get_job(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "cust-JGTaMbJMdqjJU8WbQdN9Q2",
            "created_at": "2024-12-09T04:06:28.542884",
            "updated_at": "2024-12-09T04:21:19.852832",
            "config": {"name": "meta-llama/Llama-3.1-8B-Instruct", "base_model": "meta-llama/Llama-3.1-8B-Instruct"},
            "dataset": {"name": "default/sample-basic-test"},
            "hyperparameters": {
                "finetuning_type": "lora",
                "training_type": "sft",
                "batch_size": 16,
                "epochs": 2,
                "learning_rate": 0.0001,
                "lora": {"adapter_dim": 16},
            },
            "output_model": "default/job-1234",
            "status": "completed",
            "project": "default",
        }
        mock_get.return_value = mock_response

        client = MagicMock()

        with patch.object(
            client.post_training,
            "get_job",
            return_value={
                "id": "cust-JGTaMbJMdqjJU8WbQdN9Q2",
                "status": "completed",
                "created_at": "2024-12-09T04:06:28.542884",
                "updated_at": "2024-12-09T04:21:19.852832",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "dataset_id": "sample-basic-test",
                "batch_size": 16,
                "epochs": 2,
                "learning_rate": 0.0001,
                "adapter_dim": 16,
                "output_model": "default/job-1234",
            },
        ):
            job = client.post_training.get_job("cust-JGTaMbJMdqjJU8WbQdN9Q2")

            self.assertEqual(job["id"], "cust-JGTaMbJMdqjJU8WbQdN9Q2")
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["model"], "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(job["dataset_id"], "sample-basic-test")
            self.assertEqual(job["batch_size"], 16)
            self.assertEqual(job["epochs"], 2)
            self.assertEqual(job["learning_rate"], 0.0001)
            self.assertEqual(job["adapter_dim"], 16)
            self.assertEqual(job["output_model"], "default/job-1234")

    @patch("requests.delete")
    def test_cancel_job(self, mock_delete):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_delete.return_value = mock_response

        client = MagicMock()

        with patch.object(client.post_training, "cancel_job", return_value=True):
            result = client.post_training.cancel_job("cust-JGTaMbJMdqjJU8WbQdN9Q2")

            self.assertTrue(result)

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_async_supervised_fine_tune(self, mock_post):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "id": "cust-JGTaMbJMdqjJU8WbQdN9Q2",
                "status": "created",
                "created_at": "2024-12-09T04:06:28.542884",
                "updated_at": "2024-12-09T04:06:28.542884",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "dataset_id": "sample-basic-test",
                "output_model": "default/job-1234",
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = MagicMock()

        algorithm_config = LoraFinetuningConfig(type="LoRA", adapter_dim=16)

        data_config = TrainingConfigDataConfig(dataset_id="sample-basic-test", batch_size=16)

        optimizer_config = TrainingConfigOptimizerConfig(
            lr=0.0001,
        )

        training_config = TrainingConfig(
            n_epochs=2,
            data_config=data_config,
            optimizer_config=optimizer_config,
        )

        with patch.object(
            client.post_training,
            "supervised_fine_tune_async",
            AsyncMock(
                return_value={
                    "id": "cust-JGTaMbJMdqjJU8WbQdN9Q2",
                    "status": "created",
                    "created_at": "2024-12-09T04:06:28.542884",
                    "updated_at": "2024-12-09T04:06:28.542884",
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "dataset_id": "sample-basic-test",
                    "output_model": "default/job-1234",
                }
            ),
        ):
            training_job = await client.post_training.supervised_fine_tune_async(
                job_uuid="1234",
                model="meta-llama/Llama-3.1-8B-Instruct",
                checkpoint_dir="",
                algorithm_config=algorithm_config,
                training_config=training_config,
                logger_config={},
                hyperparam_search_config={},
            )

            self.assertEqual(training_job["id"], "cust-JGTaMbJMdqjJU8WbQdN9Q2")
            self.assertEqual(training_job["status"], "created")
            self.assertEqual(training_job["model"], "meta-llama/Llama-3.1-8B-Instruct")
            self.assertEqual(training_job["dataset_id"], "sample-basic-test")

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession.post")
    async def test_inference_with_fine_tuned_model(self, mock_post):
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "id": "cmpl-123456",
                "object": "text_completion",
                "created": 1677858242,
                "model": "job-1234",
                "choices": [
                    {
                        "text": "The next GTC will take place in the middle of March, 2023.",
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112},
            }
        )
        mock_post.return_value.__aenter__.return_value = mock_response

        client = MagicMock()

        with patch.object(
            client.inference,
            "completion",
            AsyncMock(
                return_value={
                    "id": "cmpl-123456",
                    "object": "text_completion",
                    "created": 1677858242,
                    "model": "job-1234",
                    "choices": [
                        {
                            "text": "The next GTC will take place in the middle of March, 2023.",
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 12, "total_tokens": 112},
                }
            ),
        ):
            response = await client.inference.completion(
                content="When is the upcoming GTC event? GTC 2018 attracted over 8,400 attendees. Due to the COVID pandemic of 2020, GTC 2020 was converted to a digital event and drew roughly 59,000 registrants. The 2021 GTC keynote, which was streamed on YouTube on April 12, included a portion that was made with CGI using the Nvidia Omniverse real-time rendering platform. This next GTC will take place in the middle of March, 2023. Answer: ",
                stream=False,
                model_id="job-1234",
                sampling_params={
                    "max_tokens": 128,
                },
            )

            self.assertEqual(response["model"], "job-1234")
            self.assertEqual(
                response["choices"][0]["text"], "The next GTC will take place in the middle of March, 2023."
            )


if __name__ == "__main__":
    unittest.main()
