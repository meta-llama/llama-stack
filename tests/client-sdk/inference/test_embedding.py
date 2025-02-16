# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest


def test_embedding(llama_stack_client):
    emb_models = [x for x in llama_stack_client.models.list() if x.model_type == "embedding"]
    if len(emb_models) == 0:
        pytest.skip("No embedding models found")

    embedding_response = llama_stack_client.inference.embeddings(
        model_id=emb_models[0].identifier, contents=["Hello, world!", "This is a test", "Testing embeddings"]
    )
    assert embedding_response is not None
    assert len(embedding_response.embeddings) == 3
    assert len(embedding_response.embeddings[0]) == emb_models[0].metadata["embedding_dimension"]
