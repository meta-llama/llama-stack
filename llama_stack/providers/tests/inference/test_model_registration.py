# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_models.datatypes import CoreModelId

# How to run this test:
#
# pytest -v -s llama_stack/providers/tests/inference/test_model_registration.py
#   -m "meta_reference"
#   --env TOGETHER_API_KEY=<your_api_key>


class TestModelRegistration:
    @pytest.mark.asyncio
    async def test_register_unsupported_model(self, inference_stack):
        _, models_impl = inference_stack

        # Try to register a model that's too large for local inference
        with pytest.raises(Exception) as exc_info:
            await models_impl.register_model(
                model_id="Llama3.1-70B-Instruct",
            )

    @pytest.mark.asyncio
    async def test_register_nonexistent_model(self, inference_stack):
        _, models_impl = inference_stack

        # Try to register a non-existent model
        with pytest.raises(Exception) as exc_info:
            await models_impl.register_model(
                model_id="Llama3-NonExistent-Model",
            )

    @pytest.mark.asyncio
    async def test_update_model(self, inference_stack):
        _, models_impl = inference_stack

        # Register a model to update
        model_id = CoreModelId.llama3_1_8b_instruct.value
        old_model = await models_impl.register_model(model_id=model_id)

        # Update the model
        new_model_id = CoreModelId.llama3_2_3b_instruct.value
        updated_model = await models_impl.update_model(
            model_id=model_id, provider_model_id=new_model_id
        )

        # Retrieve the updated model to verify changes
        assert updated_model.provider_resource_id != old_model.provider_resource_id

        # Cleanup
        await models_impl.delete_model(model_id=model_id)
