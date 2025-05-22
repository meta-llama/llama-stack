# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import numpy as np
import pytest
import pytest_asyncio
from pymilvus import MilvusClient

from llama_stack.apis.vector_io import QueryChunksResponse
from llama_stack.providers.remote.vector_io.milvus.config import MilvusVectorIOConfig as RemoteMilvusVectorIOConfig
from llama_stack.providers.remote.vector_io.milvus.milvus import MilvusIndex

# This test is a unit test for the MilvusVectorIOAdapter class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_milvus.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

MILVUS_PROVIDER = "milvus"


@pytest_asyncio.fixture
async def milvus_index(embedding_dimension):
    config = RemoteMilvusVectorIOConfig(uri="http://localhost:19530")
    client = MilvusClient(uri=config.uri)
    index = MilvusIndex(client=client, collection_name="test_collection")
    yield index
    await index.delete()


@pytest.mark.asyncio
async def test_add_chunks(milvus_index, sample_chunks, sample_embeddings):
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)
    # Verify collection exists and has data
    assert await asyncio.to_thread(milvus_index.client.has_collection, milvus_index.collection_name)
    count = await asyncio.to_thread(
        milvus_index.client.query,
        collection_name=milvus_index.collection_name,
        filter="",
        output_fields=["count(*)"],
    )
    assert count[0]["count(*)"] == len(sample_chunks)


@pytest.mark.asyncio
async def test_query_chunks_vector(milvus_index, sample_chunks, sample_embeddings, embedding_dimension):
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    response = await milvus_index.query_vector(query_embedding, k=2, score_threshold=0.0)
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2


@pytest.mark.asyncio
async def test_query_chunks_keyword_search(milvus_index, sample_chunks, sample_embeddings):
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    query_string = "Sentence 5"
    response = await milvus_index.query_keyword(query_string=query_string, k=3, score_threshold=0.0)

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 3, f"Expected three chunks, but got {len(response.chunks)}"

    non_existent_query_str = "blablabla"
    response_no_results = await milvus_index.query_keyword(
        query_string=non_existent_query_str, k=1, score_threshold=1.0
    )

    assert isinstance(response_no_results, QueryChunksResponse)
    assert len(response_no_results.chunks) == 0, f"Expected 0 results, but got {len(response_no_results.chunks)}"


@pytest.mark.asyncio
async def test_query_chunks_keyword_search_k_greater_than_results(milvus_index, sample_chunks, sample_embeddings):
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    query_str = "Sentence 1 from document 0"  # Should match only one chunk
    response = await milvus_index.query_keyword(query_string=query_str, k=5, score_threshold=1.0)
    assert 0 < len(response.chunks) <= 4, f"Expected results between [1, 4], got {len(response.chunks)}"
    assert any("Sentence 1 from document 0" in chunk.content for chunk in response.chunks), "Expected chunk not found"
