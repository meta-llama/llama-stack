# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llama_stack.apis.vector_io import QueryChunksResponse

# Mock the entire chromadb module
chromadb_mock = MagicMock()
chromadb_mock.AsyncHttpClient = MagicMock
chromadb_mock.PersistentClient = MagicMock

# Apply the mock before importing ChromaIndex
with patch.dict("sys.modules", {"chromadb": chromadb_mock}):
    from llama_stack.providers.remote.vector_io.chroma.chroma import ChromaIndex

# This test is a unit test for the ChromaVectorIOAdapter class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_chroma.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

CHROMA_PROVIDER = "chromadb"


@pytest.fixture
async def mock_chroma_collection() -> MagicMock:
    """Create a mock Chroma collection with common method behaviors."""
    collection = MagicMock()
    collection.name = "test_collection"

    # Mock add operation
    collection.add.return_value = None

    # Mock query operation for vector search
    collection.query.return_value = {
        "distances": [[0.1, 0.2]],
        "documents": [
            [
                json.dumps({"content": "mock chunk 1", "metadata": {"document_id": "doc1"}}),
                json.dumps({"content": "mock chunk 2", "metadata": {"document_id": "doc2"}}),
            ]
        ],
    }

    # Mock delete operation
    collection.delete.return_value = None

    return collection


@pytest.fixture
async def mock_chroma_client(mock_chroma_collection):
    """Create a mock Chroma client with common method behaviors."""
    client = MagicMock()

    # Mock collection operations
    client.get_or_create_collection.return_value = mock_chroma_collection
    client.get_collection.return_value = mock_chroma_collection
    client.delete_collection.return_value = None

    return client


@pytest.fixture
async def chroma_index(mock_chroma_client, mock_chroma_collection):
    """Create a ChromaIndex with mocked client and collection."""
    index = ChromaIndex(client=mock_chroma_client, collection=mock_chroma_collection)
    yield index
    # No real cleanup needed since we're using mocks


async def test_add_chunks(chroma_index, sample_chunks, sample_embeddings, mock_chroma_collection):
    await chroma_index.add_chunks(sample_chunks, sample_embeddings)

    # Verify data was inserted
    mock_chroma_collection.add.assert_called_once()

    # Verify the add call had the right number of chunks
    add_call = mock_chroma_collection.add.call_args
    assert len(add_call[1]["documents"]) == len(sample_chunks)


async def test_query_chunks_vector(
    chroma_index, sample_chunks, sample_embeddings, embedding_dimension, mock_chroma_collection
):
    # Setup: Add chunks first
    await chroma_index.add_chunks(sample_chunks, sample_embeddings)

    # Test vector search
    query_embedding = np.random.rand(embedding_dimension).astype(np.float32)
    response = await chroma_index.query_vector(query_embedding, k=2, score_threshold=0.0)

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2
    mock_chroma_collection.query.assert_called_once()


async def test_query_chunks_keyword_search(chroma_index, sample_chunks, sample_embeddings, mock_chroma_collection):
    await chroma_index.add_chunks(sample_chunks, sample_embeddings)

    # Test keyword search
    query_string = "Sentence 5"
    response = await chroma_index.query_keyword(query_string=query_string, k=2, score_threshold=0.0)

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2


async def test_delete_collection(chroma_index, mock_chroma_client):
    # Test collection deletion
    await chroma_index.delete()

    mock_chroma_client.delete_collection.assert_called_once_with(chroma_index.collection.name)
