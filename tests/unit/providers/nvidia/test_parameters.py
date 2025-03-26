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
from llama_stack_client.types.algorithm_config_param import LoraFinetuningConfig
from llama_stack_client.types.post_training_supervised_fine_tune_params import (
    TrainingConfig,
    TrainingConfigDataConfig,
    TrainingConfigEfficiencyConfig,
    TrainingConfigOptimizerConfig,
)

from llama_stack.providers.remote.post_training.nvidia.post_training import (
    NvidiaPostTrainingAdapter,
    NvidiaPostTrainingConfig,
)


class TestNvidiaParameters(unittest.TestCase):
    def setUp(self):
        os.environ["NVIDIA_BASE_URL"] = "http://nemo.test"
        os.environ["NVIDIA_CUSTOMIZER_URL"] = "http://nemo.test"

        config = NvidiaPostTrainingConfig(
            base_url=os.environ["NVIDIA_BASE_URL"], customizer_url=os.environ["NVIDIA_CUSTOMIZER_URL"], api_key=None
        )
        self.adapter = NvidiaPostTrainingAdapter(config)

        self.make_request_patcher = patch(
            "llama_stack.providers.remote.post_training.nvidia.post_training.NvidiaPostTrainingAdapter._make_request"
        )
        self.mock_make_request = self.make_request_patcher.start()
        self.mock_make_request.return_value = {
            "id": "job-123",
            "status": "created",
            "created_at": "2025-03-04T13:07:47.543605",
            "updated_at": "2025-03-04T13:07:47.543605",
        }

    def tearDown(self):
        self.make_request_patcher.stop()

    def _assert_request_params(self, expected_json):
        """Helper method to verify parameters in the request JSON."""
        call_args = self.mock_make_request.call_args
        actual_json = call_args[1]["json"]

        for key, value in expected_json.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    assert actual_json[key][nested_key] == nested_value
            else:
                assert actual_json[key] == value

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, run_async):
        self.run_async = run_async

    def test_customizer_parameters_passed(self):
        """Test scenario 1: When an optional parameter is passed and value is correctly set."""
        custom_adapter_dim = 32  # Different from default of 8
        algorithm_config = LoraFinetuningConfig(
            type="LoRA",
            adapter_dim=custom_adapter_dim,
            adapter_dropout=0.2,
            apply_lora_to_mlp=True,
            apply_lora_to_output=True,
            alpha=16,
            rank=16,
            lora_attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        data_config = TrainingConfigDataConfig(dataset_id="test-dataset", batch_size=16)
        optimizer_config = TrainingConfigOptimizerConfig(lr=0.0002)
        training_config = TrainingConfig(
            n_epochs=3,
            data_config=data_config,
            optimizer_config=optimizer_config,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.run_async(
                self.adapter.supervised_fine_tune(
                    job_uuid="test-job",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    checkpoint_dir="",
                    algorithm_config=algorithm_config,
                    training_config=training_config,
                    logger_config={},
                    hyperparam_search_config={},
                )
            )

            warning_texts = [str(warning.message) for warning in w]

            fields = [
                "apply_lora_to_output",
                "lora_attn_modules",
                "apply_lora_to_mlp",
            ]
            for field in fields:
                assert any(field in text for text in warning_texts)

        self._assert_request_params(
            {
                "hyperparameters": {
                    "lora": {"adapter_dim": custom_adapter_dim, "adapter_dropout": 0.2, "alpha": 16},
                    "epochs": 3,
                    "learning_rate": 0.0002,
                    "batch_size": 16,
                }
            }
        )

    def test_required_parameters_passed(self):
        """Test scenario 2: When required parameters are passed."""
        required_model = "meta-llama/Llama-3.1-8B-Instruct"
        required_dataset_id = "required-dataset"
        required_job_uuid = "required-job"

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

        data_config = TrainingConfigDataConfig(
            dataset_id=required_dataset_id,  # Required parameter
            batch_size=8,
        )

        optimizer_config = TrainingConfigOptimizerConfig(lr=0.0001)

        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            optimizer_config=optimizer_config,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.run_async(
                self.adapter.supervised_fine_tune(
                    job_uuid=required_job_uuid,  # Required parameter
                    model=required_model,  # Required parameter
                    checkpoint_dir="",
                    algorithm_config=algorithm_config,
                    training_config=training_config,
                    logger_config={},
                    hyperparam_search_config={},
                )
            )

            warning_texts = [str(warning.message) for warning in w]

            fields = [
                "rank",
                "apply_lora_to_output",
                "lora_attn_modules",
                "apply_lora_to_mlp",
            ]
            for field in fields:
                assert any(field in text for text in warning_texts)

        self.mock_make_request.assert_called_once()
        call_args = self.mock_make_request.call_args

        assert call_args[1]["json"]["config"] == "meta/llama-3.1-8b-instruct"
        assert call_args[1]["json"]["dataset"]["name"] == required_dataset_id

    def test_unsupported_parameters_warning(self):
        """Test that warnings are raised for unsupported parameters."""
        data_config = TrainingConfigDataConfig(
            dataset_id="test-dataset",
            batch_size=8,
            # Unsupported parameters
            shuffle=True,
            data_format="instruct",
            validation_dataset_id="val-dataset",
        )

        optimizer_config = TrainingConfigOptimizerConfig(
            lr=0.0001,
            weight_decay=0.01,
            # Unsupported parameters
            optimizer_type="adam",
            num_warmup_steps=100,
        )

        efficiency_config = TrainingConfigEfficiencyConfig(
            enable_activation_checkpointing=True  # Unsupported parameter
        )

        training_config = TrainingConfig(
            n_epochs=1,
            data_config=data_config,
            optimizer_config=optimizer_config,
            # Unsupported parameters
            efficiency_config=efficiency_config,
            max_steps_per_epoch=1000,
            gradient_accumulation_steps=4,
            max_validation_steps=100,
            dtype="bf16",
        )

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            self.run_async(
                self.adapter.supervised_fine_tune(
                    job_uuid="test-job",
                    model="meta-llama/Llama-3.1-8B-Instruct",
                    checkpoint_dir="test-dir",  # Unsupported parameter
                    algorithm_config=LoraFinetuningConfig(
                        type="LoRA",
                        adapter_dim=16,
                        adapter_dropout=0.1,
                        apply_lora_to_mlp=True,
                        apply_lora_to_output=True,
                        alpha=16,
                        rank=16,
                        lora_attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    ),
                    training_config=training_config,
                    logger_config={"test": "value"},  # Unsupported parameter
                    hyperparam_search_config={"test": "value"},  # Unsupported parameter
                )
            )

            assert len(w) >= 4
            warning_texts = [str(warning.message) for warning in w]

            fields = [
                "checkpoint_dir",
                "hyperparam_search_config",
                "logger_config",
                "TrainingConfig",
                "DataConfig",
                "OptimizerConfig",
                "max_steps_per_epoch",
                "gradient_accumulation_steps",
                "max_validation_steps",
                "dtype",
                # required unsupported parameters
                "rank",
                "apply_lora_to_output",
                "lora_attn_modules",
                "apply_lora_to_mlp",
            ]
            for field in fields:
                assert any(field in text for text in warning_texts)


if __name__ == "__main__":
    unittest.main()
