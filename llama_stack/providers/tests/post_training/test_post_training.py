# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import pytest
from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.apis.post_training import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403

# How to run this test:
#
# pytest llama_stack/providers/tests/post_training/test_post_training.py
#   -m "torchtune_post_training_huggingface_datasetio"
#   -v -s --tb=short --disable-warnings


class TestPostTraining:
    @pytest.mark.asyncio
    async def test_supervised_fine_tune(self, post_training_stack):
        algorithm_config = LoraFinetuningConfig(
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
