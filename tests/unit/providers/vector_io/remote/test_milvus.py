# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from llama_stack.apis.vector_io import QueryChunksResponse

# Mock the entire pymilvus module
pymilvus_mock = MagicMock()
pymilvus_mock.DataType = MagicMock()
pymilvus_mock.MilvusClient = MagicMock

# Apply the mock before importing MilvusIndex
with patch.dict("sys.modules", {"pymilvus": pymilvus_mock}):
    from llama_stack.providers.remote.vector_io.milvus.milvus import MilvusIndex, MilvusVectorIOAdapter
    from llama_stack.providers.remote.vector_io.milvus.config import MilvusVectorIOConfig as RemoteMilvusVectorIOConfig

# This test is a unit test for the MilvusVectorIOAdapter class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_milvus.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

MILVUS_PROVIDER = "milvus"


@pytest.fixture
async def mock_milvus_client() -> MagicMock:
    """Create a mock Milvus client with common method behaviors."""
    client = MagicMock()

    # Mock collection operations
    client.has_collection.return_value = False  # Initially no collection
    client.create_collection.return_value = None
    client.drop_collection.return_value = None

    # Mock insert operation
    client.insert.return_value = {"insert_count": 10}

    # Mock search operation - return mock results (data should be dict, not JSON string)
    client.search.return_value = [
        [
            {
                "id": 0,
                "distance": 0.1,
                "entity": {"chunk_content": {"content": "mock chunk 1", "metadata": {"document_id": "doc1"}}},
            },
            {
                "id": 1,
                "distance": 0.2,
                "entity": {"chunk_content": {"content": "mock chunk 2", "metadata": {"document_id": "doc2"}}},
            },
        ]
    ]

    # Mock query operation for keyword search (data should be dict, not JSON string)
    client.query.return_value = [
        {
            "chunk_id": "chunk1",
            "chunk_content": {"content": "mock chunk 1", "metadata": {"document_id": "doc1"}},
            "score": 0.9,
        },
        {
            "chunk_id": "chunk2",
            "chunk_content": {"content": "mock chunk 2", "metadata": {"document_id": "doc2"}},
            "score": 0.8,
        },
        {
            "chunk_id": "chunk3",
            "chunk_content": {"content": "mock chunk 3", "metadata": {"document_id": "doc3"}},
            "score": 0.7,
        },
    ]

    return client


@pytest.fixture
async def milvus_index(mock_milvus_client):
    """Create a MilvusIndex with mocked client."""
    index = MilvusIndex(client=mock_milvus_client, collection_name="test_collection")
    yield index
    # No real cleanup needed since we're using mocks


@pytest.fixture
async def mock_inference_api():
    """Create a mock inference API."""
    api = MagicMock()
    api.embed.return_value = np.array([[0.1, 0.2, 0.3]])
    return api


@pytest.fixture
async def remote_milvus_config_with_kvstore():
    """Create a remote Milvus config with kvstore."""
    from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig
    
    config = RemoteMilvusVectorIOConfig(
        uri="http://localhost:19530",
        token=None,
        consistency_level="Strong",
        kvstore=SqliteKVStoreConfig(db_path="/tmp/test.db"),  # Use proper kvstore config
    )
    return config


@pytest.fixture
async def remote_milvus_config_without_kvstore():
    """Create a remote Milvus config without kvstore (None)."""
    config = RemoteMilvusVectorIOConfig(
        uri="http://localhost:19530",
        token=None,
        consistency_level="Strong",
        kvstore=None,  # No kvstore
    )
    return config


async def test_add_chunks(milvus_index, sample_chunks, sample_embeddings, mock_milvus_client):
    # Setup: collection doesn't exist initially, then exists after creation
    mock_milvus_client.has_collection.side_effect = [False, True]

    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Verify collection was created and data was inserted
    mock_milvus_client.create_collection.assert_called_once()
    mock_milvus_client.insert.assert_called_once()

    # Verify the data format in the insert call
    insert_call = mock_milvus_client.insert.call_args
    assert insert_call[1]["collection_name"] == "test_collection"
    assert len(insert_call[1]["data"]) == len(sample_chunks)


async def test_query_chunks_vector(
    milvus_index, sample_chunks, sample_embeddings, embedding_dimension, mock_milvus_client
):
    # Setup: Add chunks first
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Query with a test embedding
    query_embedding = np.random.rand(embedding_dimension)
    response = await milvus_index.query_vector(query_embedding, k=2, score_threshold=0.0)

    # Verify search was called and response is valid
    mock_milvus_client.search.assert_called_once()
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2


async def test_query_chunks_keyword_search(milvus_index, sample_chunks, sample_embeddings, mock_milvus_client):
    # Setup: Add chunks first
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Test keyword search
    query_string = "test query"
    response = await milvus_index.query_keyword(query_string=query_string, k=2, score_threshold=0.0)

    # Verify search was called and response is valid
    mock_milvus_client.search.assert_called_once()
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2


async def test_bm25_fallback_to_simple_search(milvus_index, sample_chunks, sample_embeddings, mock_milvus_client):
    # Setup: Add chunks first
    mock_milvus_client.has_collection.return_value = True
    await milvus_index.add_chunks(sample_chunks, sample_embeddings)

    # Mock BM25 search to fail, triggering fallback
    mock_milvus_client.search.side_effect = Exception("BM25 search not available")

    # Mock the fallback query to return results
    mock_milvus_client.query.return_value = [
        {
            "chunk_id": "chunk1",
            "chunk_content": {"content": "mock chunk 1", "metadata": {"document_id": "doc1"}},
        },
        {
            "chunk_id": "chunk2",
            "chunk_content": {"content": "mock chunk 2", "metadata": {"document_id": "doc2"}},
        },
        {
            "chunk_id": "chunk3",
            "chunk_content": {"content": "mock chunk 3", "metadata": {"document_id": "doc3"}},
        },
    ]

    # Test keyword search with fallback
    query_string = "test query"
    response = await milvus_index.query_keyword(query_string=query_string, k=3, score_threshold=0.0)

    # Verify both search and query were called (search failed, query succeeded)
    mock_milvus_client.query.assert_called_once()
    mock_milvus_client.search.assert_called_once()  # Called once but failed

    # Verify the query call arguments
    query_call_args = mock_milvus_client.query.call_args
    assert query_call_args[1]["collection_name"] == "test_collection"
    assert "content like" in query_call_args[1]["filter"]

    # Verify response is valid
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 3


async def test_delete_collection(milvus_index, mock_milvus_client):
    # Test collection deletion
    mock_milvus_client.has_collection.return_value = True

    await milvus_index.delete()

    mock_milvus_client.drop_collection.assert_called_once_with(collection_name=milvus_index.collection_name)


# Tests for kvstore None handling fix
async def test_remote_milvus_initialization_with_kvstore(remote_milvus_config_with_kvstore, mock_inference_api):
    """Test that remote Milvus initializes correctly with kvstore configured."""
    with patch("llama_stack.providers.remote.vector_io.milvus.milvus.MilvusClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        with patch("llama_stack.providers.remote.vector_io.milvus.milvus.kvstore_impl") as mock_kvstore_impl:
            mock_kvstore = MagicMock()
            mock_kvstore_impl.return_value = mock_kvstore
            mock_kvstore.values_in_range.return_value = asyncio.Future()
            mock_kvstore.values_in_range.return_value.set_result([])
            mock_kvstore.set.return_value = asyncio.Future()
            mock_kvstore.set.return_value.set_result(None)
            
            adapter = MilvusVectorIOAdapter(
                config=remote_milvus_config_with_kvstore,
                inference_api=mock_inference_api,
                files_api=None,
            )
            
            await adapter.initialize()
            
            # Verify kvstore was initialized
            mock_kvstore_impl.assert_called_once_with(remote_milvus_config_with_kvstore.kvstore)
            assert adapter.kvstore is not None


async def test_remote_milvus_initialization_without_kvstore(remote_milvus_config_without_kvstore, mock_inference_api):
    """Test that remote Milvus initializes correctly without kvstore (None)."""
    with patch("llama_stack.providers.remote.vector_io.milvus.milvus.MilvusClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        adapter = MilvusVectorIOAdapter(
            config=remote_milvus_config_without_kvstore,
            inference_api=mock_inference_api,
            files_api=None,
        )
        
        await adapter.initialize()
        
        # Verify kvstore is None and no kvstore_impl was called
        assert adapter.kvstore is None


async def test_openai_vector_store_methods_without_kvstore(remote_milvus_config_without_kvstore, mock_inference_api):
    """Test that OpenAI vector store methods work correctly when kvstore is None."""
    with patch("llama_stack.providers.remote.vector_io.milvus.milvus.MilvusClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        adapter = MilvusVectorIOAdapter(
            config=remote_milvus_config_without_kvstore,
            inference_api=mock_inference_api,
            files_api=None,
        )
        
        await adapter.initialize()
        
        # Test _save_openai_vector_store with None kvstore
        store_id = "test_store"
        store_info = {"id": store_id, "name": "test"}
        
        # Should not raise an error
        await adapter._save_openai_vector_store(store_id, store_info)
        
        # Verify store was added to in-memory cache
        assert store_id in adapter.openai_vector_stores
        assert adapter.openai_vector_stores[store_id] == store_info


async def test_openai_vector_store_methods_with_kvstore(remote_milvus_config_with_kvstore, mock_inference_api):
    """Test that OpenAI vector store methods work correctly when kvstore is configured."""
    with patch("llama_stack.providers.remote.vector_io.milvus.milvus.MilvusClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        with patch("llama_stack.providers.remote.vector_io.milvus.milvus.kvstore_impl") as mock_kvstore_impl:
            mock_kvstore = MagicMock()
            mock_kvstore_impl.return_value = mock_kvstore
            mock_kvstore.values_in_range.return_value = asyncio.Future()
            mock_kvstore.values_in_range.return_value.set_result([])
            mock_kvstore.set.return_value = asyncio.Future()
            mock_kvstore.set.return_value.set_result(None)
            
            adapter = MilvusVectorIOAdapter(
                config=remote_milvus_config_with_kvstore,
                inference_api=mock_inference_api,
                files_api=None,
            )
            
            await adapter.initialize()
            
            # Test _save_openai_vector_store with kvstore
            store_id = "test_store"
            store_info = {"id": store_id, "name": "test"}
            
            await adapter._save_openai_vector_store(store_id, store_info)
            
            # Verify both kvstore and in-memory cache were updated
            mock_kvstore.set.assert_called_once()
            assert store_id in adapter.openai_vector_stores
            assert adapter.openai_vector_stores[store_id] == store_info


async def test_load_openai_vector_stores_without_kvstore(remote_milvus_config_without_kvstore, mock_inference_api):
    """Test that _load_openai_vector_stores returns empty dict when kvstore is None."""
    with patch("llama_stack.providers.remote.vector_io.milvus.milvus.MilvusClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        adapter = MilvusVectorIOAdapter(
            config=remote_milvus_config_without_kvstore,
            inference_api=mock_inference_api,
            files_api=None,
        )
        
        await adapter.initialize()
        
        # Should return empty dict when kvstore is None
        result = await adapter._load_openai_vector_stores()
        assert result == {}


async def test_delete_openai_vector_store_without_kvstore(remote_milvus_config_without_kvstore, mock_inference_api):
    """Test that _delete_openai_vector_store_from_storage works when kvstore is None."""
    with patch("llama_stack.providers.remote.vector_io.milvus.milvus.MilvusClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        adapter = MilvusVectorIOAdapter(
            config=remote_milvus_config_without_kvstore,
            inference_api=mock_inference_api,
            files_api=None,
        )
        
        await adapter.initialize()
        
        # Add a store to in-memory cache
        store_id = "test_store"
        adapter.openai_vector_stores[store_id] = {"id": store_id}
        
        # Should not raise an error and should clean up in-memory cache
        await adapter._delete_openai_vector_store_from_storage(store_id)
        
        # Verify store was removed from in-memory cache
        assert store_id not in adapter.openai_vector_stores
