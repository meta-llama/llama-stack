# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llama_stack.apis.models import Model, ModelType
from llama_stack.apis.post_training.post_training import (
    DataConfig,
    DatasetFormat,
    LoraFinetuningConfig,
    OptimizerConfig,
    OptimizerType,
    QATFinetuningConfig,
    TrainingConfig,
)
from llama_stack.distribution.library_client import convert_pydantic_to_json_value
from llama_stack.providers.remote.inference.nvidia.nvidia import NVIDIAConfig, NVIDIAInferenceAdapter
from llama_stack.providers.remote.post_training.nvidia.post_training import (
    ListNvidiaPostTrainingJobs,
    NvidiaPostTrainingAdapter,
    NvidiaPostTrainingConfig,
    NvidiaPostTrainingJob,
    NvidiaPostTrainingJobStatusResponse,
)


@pytest.fixture
def nvidia_adapters():
    """Set up the NVIDIA adapters with mocked dependencies"""
    os.environ["NVIDIA_BASE_URL"] = "http://nemo.test"  # needed for llm inference
    os.environ["NVIDIA_CUSTOMIZER_URL"] = "http://nemo.test"  # needed for nemo customizer

    config = NvidiaPostTrainingConfig(
        base_url=os.environ["NVIDIA_BASE_URL"], customizer_url=os.environ["NVIDIA_CUSTOMIZER_URL"], api_key=None
    )
    adapter = NvidiaPostTrainingAdapter(config)

    # Mock the inference client
    inference_config = NVIDIAConfig(base_url=os.environ["NVIDIA_BASE_URL"], api_key=None)
    inference_adapter = NVIDIAInferenceAdapter(inference_config)

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock()

    with (
        patch(
            "llama_stack.providers.remote.post_training.nvidia.post_training.NvidiaPostTrainingAdapter._make_request"
        ) as mock_make_request,
        patch(
            "llama_stack.providers.remote.inference.nvidia.nvidia.NVIDIAInferenceAdapter._get_client",
            return_value=mock_client,
        ),
    ):
        yield adapter, inference_adapter, mock_make_request, mock_client


def assert_request(mock_call, expected_method, expected_path, expected_params=None, expected_json=None):
    """Helper function to verify request details in mock calls."""
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


async def test_supervised_fine_tune(nvidia_adapters):
    """Test the supervised fine-tuning API call."""
    adapter, inference_adapter, mock_make_request, mock_client = nvidia_adapters

    mock_make_request.return_value = {
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
            "lora": {"alpha": 16},
        },
        "output_model": "default/job-1234",
        "status": "created",
        "project": "default",
        "custom_fields": {},
        "ownership": {"created_by": "me", "access_policies": {}},
    }

    algorithm_config = LoraFinetuningConfig(
        type="LoRA",
        apply_lora_to_mlp=True,
        apply_lora_to_output=True,
        alpha=16,
        rank=16,
        lora_attn_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    data_config = DataConfig(
        dataset_id="sample-basic-test", batch_size=16, shuffle=False, data_format=DatasetFormat.instruct
    )

    optimizer_config = OptimizerConfig(
        optimizer_type=OptimizerType.adam,
        lr=0.0001,
        weight_decay=0.01,
        num_warmup_steps=100,
    )

    training_config = TrainingConfig(
        n_epochs=2,
        data_config=data_config,
        optimizer_config=optimizer_config,
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        training_job = await adapter.supervised_fine_tune(
            job_uuid="1234",
            model="meta/llama-3.2-1b-instruct@v1.0.0+L40",
            checkpoint_dir="",
            algorithm_config=algorithm_config,
            training_config=convert_pydantic_to_json_value(training_config),
            logger_config={},
            hyperparam_search_config={},
        )

    # check the output is a PostTrainingJob
    assert isinstance(training_job, NvidiaPostTrainingJob)
    assert training_job.job_uuid == "cust-JGTaMbJMdqjJU8WbQdN9Q2"

    mock_make_request.assert_called_once()
    assert_request(
        mock_make_request,
        "POST",
        "/v1/customization/jobs",
        expected_json={
            "config": "meta/llama-3.2-1b-instruct@v1.0.0+L40",
            "dataset": {"name": "sample-basic-test", "namespace": "default"},
            "hyperparameters": {
                "training_type": "sft",
                "finetuning_type": "lora",
                "epochs": 2,
                "batch_size": 16,
                "learning_rate": 0.0001,
                "weight_decay": 0.01,
                "lora": {"alpha": 16},
            },
        },
    )


async def test_supervised_fine_tune_with_qat(nvidia_adapters):
    adapter, inference_adapter, mock_make_request, mock_client = nvidia_adapters

    algorithm_config = QATFinetuningConfig(type="QAT", quantizer_name="quantizer_name", group_size=1)
    data_config = DataConfig(
        dataset_id="sample-basic-test", batch_size=16, shuffle=False, data_format=DatasetFormat.instruct
    )
    optimizer_config = OptimizerConfig(
        optimizer_type=OptimizerType.adam,
        lr=0.0001,
        weight_decay=0.01,
        num_warmup_steps=100,
    )
    training_config = TrainingConfig(
        n_epochs=2,
        data_config=data_config,
        optimizer_config=optimizer_config,
    )
    # This will raise NotImplementedError since QAT is not supported
    with pytest.raises(NotImplementedError):
        await adapter.supervised_fine_tune(
            job_uuid="1234",
            model="meta/llama-3.2-1b-instruct@v1.0.0+L40",
            checkpoint_dir="",
            algorithm_config=algorithm_config,
            training_config=convert_pydantic_to_json_value(training_config),
            logger_config={},
            hyperparam_search_config={},
        )


@pytest.mark.parametrize(
    "customizer_status,expected_status",
    [
        ("running", "in_progress"),
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("pending", "scheduled"),
        ("unknown", "scheduled"),
    ],
)
async def test_get_training_job_status(nvidia_adapters, customizer_status, expected_status):
    adapter, inference_adapter, mock_make_request, mock_client = nvidia_adapters

    mock_make_request.return_value = {
        "created_at": "2024-12-09T04:06:28.580220",
        "updated_at": "2024-12-09T04:21:19.852832",
        "status": customizer_status,
        "steps_completed": 1210,
        "epochs_completed": 2,
        "percentage_done": 100.0,
        "best_epoch": 2,
        "train_loss": 1.718016266822815,
        "val_loss": 1.8661999702453613,
    }

    job_id = "cust-JGTaMbJMdqjJU8WbQdN9Q2"

    status = await adapter.get_training_job_status(job_uuid=job_id)

    assert isinstance(status, NvidiaPostTrainingJobStatusResponse)
    assert status.status.value == expected_status
    assert status.steps_completed == 1210
    assert status.epochs_completed == 2
    assert status.percentage_done == 100.0
    assert status.best_epoch == 2
    assert status.train_loss == 1.718016266822815
    assert status.val_loss == 1.8661999702453613

    assert_request(
        mock_make_request,
        "GET",
        f"/v1/customization/jobs/{job_id}/status",
        expected_params={"job_id": job_id},
    )


async def test_get_training_jobs(nvidia_adapters):
    adapter, inference_adapter, mock_make_request, mock_client = nvidia_adapters

    job_id = "cust-JGTaMbJMdqjJU8WbQdN9Q2"
    mock_make_request.return_value = {
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

    jobs = await adapter.get_training_jobs()

    assert isinstance(jobs, ListNvidiaPostTrainingJobs)
    assert len(jobs.data) == 1
    job = jobs.data[0]
    assert job.job_uuid == job_id
    assert job.status.value == "completed"

    mock_make_request.assert_called_once()
    assert_request(
        mock_make_request,
        "GET",
        "/v1/customization/jobs",
        expected_params={"page": 1, "page_size": 10, "sort": "created_at"},
    )


async def test_cancel_training_job(nvidia_adapters):
    adapter, inference_adapter, mock_make_request, mock_client = nvidia_adapters

    mock_make_request.return_value = {}  # Empty response for successful cancellation
    job_id = "cust-JGTaMbJMdqjJU8WbQdN9Q2"

    result = await adapter.cancel_training_job(job_uuid=job_id)

    assert result is None

    mock_make_request.assert_called_once()
    assert_request(
        mock_make_request,
        "POST",
        f"/v1/customization/jobs/{job_id}/cancel",
        expected_params={"job_id": job_id},
    )


async def test_inference_register_model(nvidia_adapters):
    adapter, inference_adapter, mock_make_request, mock_client = nvidia_adapters

    model_id = "default/job-1234"
    model_type = ModelType.llm
    model = Model(
        identifier=model_id,
        provider_id="nvidia",
        provider_model_id=model_id,
        provider_resource_id=model_id,
        model_type=model_type,
    )
    result = await inference_adapter.register_model(model)
    assert result == model
    assert len(inference_adapter.alias_to_provider_id_map) > 1
    assert inference_adapter.get_provider_model_id(model.provider_model_id) == model_id

    with patch.object(inference_adapter, "chat_completion") as mock_chat_completion:
        await inference_adapter.chat_completion(
            model_id=model_id,
            messages=[{"role": "user", "content": "Hello, model"}],
        )

        mock_chat_completion.assert_called()
