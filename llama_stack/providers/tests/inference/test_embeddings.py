# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.inference import EmbeddingsResponse, ModelType

# How to run this test:
# pytest -v -s llama_stack/providers/tests/inference/test_embeddings.py


class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_embeddings(self, inference_model, inference_stack):
        inference_impl, models_impl = inference_stack
        model = await models_impl.get_model(inference_model)

        if model.model_type != ModelType.embedding:
            pytest.skip("This test is only applicable for embedding models")

        response = await inference_impl.embeddings(
            model_id=inference_model,
            contents=["Hello, world!"],
        )
        assert isinstance(response, EmbeddingsResponse)
        assert len(response.embeddings) > 0
        assert all(isinstance(embedding, list) for embedding in response.embeddings)
        assert all(
            isinstance(value, float)
            for embedding in response.embeddings
            for value in embedding
        )

    @pytest.mark.asyncio
    async def test_batch_embeddings(self, inference_model, inference_stack):
        inference_impl, models_impl = inference_stack
        model = await models_impl.get_model(inference_model)

        if model.model_type != ModelType.embedding:
            pytest.skip("This test is only applicable for embedding models")

        texts = ["Hello, world!", "This is a test", "Testing embeddings"]

        response = await inference_impl.embeddings(
            model_id=inference_model,
            contents=texts,
        )

        assert isinstance(response, EmbeddingsResponse)
        assert len(response.embeddings) == len(texts)
        assert all(isinstance(embedding, list) for embedding in response.embeddings)
        assert all(
            isinstance(value, float)
            for embedding in response.embeddings
            for value in embedding
        )

        embedding_dim = len(response.embeddings[0])
        assert all(len(embedding) == embedding_dim for embedding in response.embeddings)
