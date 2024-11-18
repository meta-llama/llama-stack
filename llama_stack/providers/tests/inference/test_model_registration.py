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
    async def test_register_unsupported_model(self, inference_stack, inference_model):
        inference_impl, models_impl = inference_stack

        provider = inference_impl.routing_table.get_provider_impl(inference_model)
        if provider.__provider_spec__.provider_type not in (
            "meta-reference",
            "remote::ollama",
            "remote::vllm",
            "remote::tgi",
        ):
            pytest.skip("70B instruct is too big only for local inference providers")

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
    async def test_register_with_llama_model(self, inference_stack):
        _, models_impl = inference_stack

        _ = await models_impl.register_model(
            model_id="custom-model",
            metadata={"llama_model": CoreModelId.llama3_1_8b_instruct.value},
        )
