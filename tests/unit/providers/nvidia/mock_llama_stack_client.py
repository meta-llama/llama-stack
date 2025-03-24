# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from llama_stack_client.types.algorithm_config_param import QatFinetuningConfig
from llama_stack_client.types.post_training.job_status_response import JobStatusResponse
from llama_stack_client.types.post_training_job import PostTrainingJob


class MockLlamaStackClient:
    """Mock client for testing NVIDIA post-training functionality."""

    def __init__(self, provider="nvidia"):
        self.provider = provider
        self.post_training = MockPostTraining()
        self.inference = MockInference()
        self._session = None

    def initialize(self):
        """Mock initialization method."""
        return True

    def close(self):
        """Close any open resources."""
        pass


class MockPostTraining:
    """Mock post-training module."""

    def __init__(self):
        self.job = MockPostTrainingJob()

    def supervised_fine_tune(
        self,
        job_uuid,
        model,
        checkpoint_dir,
        algorithm_config,
        training_config,
        logger_config,
        hyperparam_search_config,
    ):
        """Mock supervised fine-tuning method."""
        if isinstance(algorithm_config, QatFinetuningConfig):
            raise NotImplementedError("QAT fine-tuning is not supported by NVIDIA provider")

        # Return a mock PostTrainingJob
        return PostTrainingJob(
            job_uuid="cust-JGTaMbJMdqjJU8WbQdN9Q2",
            status="created",
            created_at="2024-12-09T04:06:28.542884",
            updated_at="2024-12-09T04:06:28.542884",
            model=model,
            dataset_id=training_config.data_config.dataset_id,
            output_model="default/job-1234",
        )

    async def supervised_fine_tune_async(
        self,
        job_uuid,
        model,
        checkpoint_dir,
        algorithm_config,
        training_config,
        logger_config,
        hyperparam_search_config,
    ):
        """Mock async supervised fine-tuning method."""
        if isinstance(algorithm_config, QatFinetuningConfig):
            raise NotImplementedError("QAT fine-tuning is not supported by NVIDIA provider")

        # Return a mock response dictionary
        return {
            "job_uuid": "cust-JGTaMbJMdqjJU8WbQdN9Q2",
            "status": "created",
            "created_at": "2024-12-09T04:06:28.542884",
            "updated_at": "2024-12-09T04:06:28.542884",
            "model": model,
            "dataset_id": training_config.data_config.dataset_id,
            "output_model": "default/job-1234",
        }


class MockPostTrainingJob:
    """Mock post-training job module."""

    def status(self, job_uuid):
        """Mock job status method."""
        return JobStatusResponse(
            status="completed",
            steps_completed=1210,
            epochs_completed=2,
            percentage_done=100.0,
            best_epoch=2,
            train_loss=1.718016266822815,
            val_loss=1.8661999702453613,
        )

    def list(self):
        """Mock job list method."""
        return [
            PostTrainingJob(
                job_uuid="cust-JGTaMbJMdqjJU8WbQdN9Q2",
                status="completed",
                created_at="2024-12-09T04:06:28.542884",
                updated_at="2024-12-09T04:21:19.852832",
                model="meta-llama/Llama-3.1-8B-Instruct",
                dataset_id="sample-basic-test",
                output_model="default/job-1234",
            )
        ]

    def cancel(self, job_uuid):
        """Mock job cancel method."""
        return None


class MockInference:
    """Mock inference module."""

    async def completion(
        self,
        content,
        stream=False,
        model_id=None,
        sampling_params=None,
    ):
        """Mock completion method."""
        return {
            "id": "cmpl-123456",
            "object": "text_completion",
            "created": 1677858242,
            "model": model_id,
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
