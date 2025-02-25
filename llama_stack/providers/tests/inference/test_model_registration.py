# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

# How to run this test:
#
# torchrun $CONDA_PREFIX/bin/pytest -v -s -k "meta_reference" --inference-model="Llama3.1-8B-Instruct"
#  ./llama_stack/providers/tests/inference/test_model_registration.py


class TestModelRegistration:
    def provider_supports_custom_names(self, provider) -> bool:
        return "remote::ollama" not in provider.__provider_spec__.provider_type

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
            pytest.skip(
                "Skipping test for remote inference providers since they can handle large models like 70B instruct"
            )

        # Try to register a model that's too large for local inference
        with pytest.raises(ValueError):
            await models_impl.register_model(
                model_id="Llama3.1-70B-Instruct",
            )

    @pytest.mark.asyncio
    async def test_register_nonexistent_model(self, inference_stack):
        _, models_impl = inference_stack

        # Try to register a non-existent model
        with pytest.raises(ValueError):
            await models_impl.register_model(
                model_id="Llama3-NonExistent-Model",
            )

    @pytest.mark.asyncio
    async def test_register_with_llama_model(self, inference_stack, inference_model):
        inference_impl, models_impl = inference_stack
        provider = inference_impl.routing_table.get_provider_impl(inference_model)
        if not self.provider_supports_custom_names(provider):
            pytest.skip("Provider does not support custom model names")

        _, models_impl = inference_stack

        _ = await models_impl.register_model(
            model_id="custom-model",
            metadata={
                "llama_model": "meta-llama/Llama-2-7b",
                "skip_load": True,
            },
        )

        with pytest.raises(ValueError):
            await models_impl.register_model(
                model_id="custom-model-2",
                metadata={
                    "llama_model": "meta-llama/Llama-2-7b",
                },
                provider_model_id="custom-model",
            )

    @pytest.mark.asyncio
    async def test_register_with_invalid_llama_model(self, inference_stack):
        _, models_impl = inference_stack

        with pytest.raises(ValueError):
            await models_impl.register_model(
                model_id="custom-model-2",
                metadata={"llama_model": "invalid-llama-model"},
            )
