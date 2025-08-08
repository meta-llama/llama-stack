# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, ChunkMetadata, QueryChunksResponse
from llama_stack.providers.remote.vector_io.pgvector.config import PGVectorVectorIOConfig
from llama_stack.providers.remote.vector_io.pgvector.pgvector import PGVectorIndex, PGVectorVectorIOAdapter

PGVECTOR_PROVIDER = "pgvector"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
def mock_psycopg2_connection():
    """Create a mock psycopg2 connection for testing."""
    connection = MagicMock()
    cursor = MagicMock()

    # Mock the cursor context manager
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock()

    # Mock connection cursor method
    connection.cursor.return_value = cursor

    return connection, cursor


@pytest.fixture
async def pgvector_index(embedding_dimension, mock_psycopg2_connection):
    """Create a PGVectorIndex instance with mocked database connection."""
    connection, cursor = mock_psycopg2_connection

    vector_db = VectorDB(
        identifier="test-vector-db",
        embedding_model="test-model",
        embedding_dimension=embedding_dimension,
        provider_id=PGVECTOR_PROVIDER,
        provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
    )

    with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
        index = PGVectorIndex(vector_db, embedding_dimension, connection)

    return index, cursor


def create_sample_chunks():
    """Create sample chunks for testing."""
    return [
        Chunk(
            content="Machine learning is a subset of artificial intelligence",
            metadata={"document_id": "doc-1", "topic": "AI"},
            chunk_metadata=ChunkMetadata(document_id="doc-1", chunk_id="chunk-1"),
        ),
        Chunk(
            content="Deep learning uses neural networks with multiple layers",
            metadata={"document_id": "doc-2", "topic": "AI"},
            chunk_metadata=ChunkMetadata(document_id="doc-2", chunk_id="chunk-2"),
        ),
        Chunk(
            content="Natural language processing enables computers to understand text",
            metadata={"document_id": "doc-3", "topic": "NLP"},
            chunk_metadata=ChunkMetadata(document_id="doc-3", chunk_id="chunk-3"),
        ),
    ]


def create_sample_embeddings(num_chunks, dimension=384):
    """Create sample embeddings for testing."""
    np.random.seed(42)
    return np.array([np.random.rand(dimension).astype(np.float32) for _ in range(num_chunks)])


class TestPGVectorIndex:
    """Test cases for PGVectorIndex class."""

    async def test_add_chunks(self, pgvector_index, embedding_dimension):
        """Test adding chunks to the index."""
        index, cursor = pgvector_index
        chunks = create_sample_chunks()
        embeddings = create_sample_embeddings(len(chunks), embedding_dimension)

        # Mock execute_values function which is used for the actual INSERT
        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.execute_values") as mock_execute_values:
            await index.add_chunks(chunks, embeddings)

            # Verify that execute_values was called with INSERT statement
            assert mock_execute_values.called
            call_args = mock_execute_values.call_args

            # Second argument is the query
            query_arg = str(call_args[0][1])
            assert "INSERT INTO" in query_arg
            assert "content_tsvector" in query_arg

    async def test_query_vector(self, pgvector_index, embedding_dimension):
        """Test vector similarity search."""
        index, cursor = pgvector_index
        query_embedding = create_sample_embeddings(1, embedding_dimension)[0]

        # Mock database response
        mock_results = [
            ({"content": "test content", "metadata": {}}, 0.1),
            ({"content": "test content 2", "metadata": {}}, 0.2),
        ]
        cursor.fetchall.return_value = mock_results

        response = await index.query_vector(query_embedding, k=2, score_threshold=0.0)

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) == 2
        assert len(response.scores) == 2

        # Verify SQL contains vector similarity search elements
        # In this case, we are using the L2 (Euclidean distance) distance metric
        call_args = cursor.execute.call_args
        assert "<->" in str(call_args) or "ORDER BY" in str(call_args)

    async def test_query_keyword(self, pgvector_index):
        """Test keyword-based full-text search."""
        index, cursor = pgvector_index
        query_string = "machine learning"

        # Mock database response
        mock_results = [
            ({"content": "Machine learning is great", "metadata": {}}, 0.8),
            ({"content": "Learning machines are useful", "metadata": {}}, 0.6),
        ]
        cursor.fetchall.return_value = mock_results

        response = await index.query_keyword(query_string, k=2, score_threshold=0.0)

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) == 2
        assert len(response.scores) == 2

        # Verify SQL contains full-text search elements
        call_args = cursor.execute.call_args
        assert "ts_rank" in str(call_args) or "plainto_tsquery" in str(call_args)

    async def test_query_keyword_with_score_threshold(self, pgvector_index):
        """Test keyword search with score filtering."""
        index, cursor = pgvector_index
        query_string = "machine learning"
        score_threshold = 0.7

        # Mock database response with mixed scores
        mock_results = [
            ({"content": "Machine learning is great", "metadata": {}}, 0.8),  # Above threshold
            ({"content": "Learning machines are useful", "metadata": {}}, 0.5),  # Below threshold
        ]
        cursor.fetchall.return_value = mock_results

        response = await index.query_keyword(query_string, k=2, score_threshold=score_threshold)

        # Should only return chunks above threshold
        assert len(response.chunks) == 1
        assert response.scores[0] >= score_threshold

    async def test_query_hybrid_rrf(self, pgvector_index, embedding_dimension):
        """Test hybrid search with RRF reranking."""
        index, cursor = pgvector_index
        query_embedding = create_sample_embeddings(1, embedding_dimension)[0]
        query_string = "machine learning"

        # Mock responses for both vector and keyword searches
        vector_results = [
            ({"content": "Vector result 1", "metadata": {}, "chunk_id": "chunk-1"}, 0.9),
            ({"content": "Vector result 2", "metadata": {}, "chunk_id": "chunk-2"}, 0.7),
        ]
        keyword_results = [
            ({"content": "Keyword result 1", "metadata": {}, "chunk_id": "chunk-3"}, 0.8),
            ({"content": "Vector result 1", "metadata": {}, "chunk_id": "chunk-1"}, 0.6),  # Overlap
        ]

        cursor.fetchall.side_effect = [vector_results, keyword_results]

        response = await index.query_hybrid(
            query_embedding,
            query_string,
            k=3,
            score_threshold=0.0,
            reranker_type="rrf",
            reranker_params={"impact_factor": 60.0},
        )

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) >= 1  # At least the overlapping chunk

        # Verify both vector and keyword searches were called
        assert cursor.execute.call_count >= 2

    async def test_query_hybrid_weighted(self, pgvector_index, embedding_dimension):
        """Test hybrid search with weighted reranking."""
        index, cursor = pgvector_index
        query_embedding = create_sample_embeddings(1, embedding_dimension)[0]
        query_string = "machine learning"

        # Mock responses
        vector_results = [
            ({"content": "Vector result", "metadata": {}, "chunk_id": "chunk-1"}, 0.9),
        ]
        keyword_results = [
            ({"content": "Keyword result", "metadata": {}, "chunk_id": "chunk-2"}, 0.8),
        ]

        cursor.fetchall.side_effect = [vector_results, keyword_results]

        response = await index.query_hybrid(
            query_embedding,
            query_string,
            k=2,
            score_threshold=0.0,
            reranker_type="weighted",
            reranker_params={"alpha": 0.7},
        )

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) == 2

    async def test_delete_chunk(self, pgvector_index):
        """Test deleting a specific chunk."""
        index, cursor = pgvector_index
        chunk_id = "test-chunk-id"

        await index.delete_chunk(chunk_id)

        # Verify DELETE query was executed
        cursor.execute.assert_called()
        call_args = cursor.execute.call_args
        assert "DELETE FROM" in str(call_args)
        assert chunk_id in str(call_args)

    async def test_delete_index(self, pgvector_index):
        """Test deleting the entire index."""
        index, cursor = pgvector_index

        await index.delete()

        # Verify DROP TABLE query was executed
        cursor.execute.assert_called_with(f"DROP TABLE IF EXISTS {index.table_name}")


class TestPGVectorVectorIOAdapter:
    """Test cases for PGVectorVectorIOAdapter class."""

    @pytest.fixture
    async def pgvector_adapter(self):
        """Create a PGVectorVectorIOAdapter instance with mocked dependencies."""
        config = PGVectorVectorIOConfig(
            host="localhost",
            port=5432,
            db="test_db",
            user="test_user",
            password="test_password",
            kvstore={"type": "sqlite", "config": {"db_path": ":memory:"}},
        )

        inference_api = AsyncMock()
        files_api = AsyncMock()

        adapter = PGVectorVectorIOAdapter(config, inference_api, files_api)

        # Mock the connection
        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
            mock_cursor.__exit__ = MagicMock()
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn

            with patch.object(adapter, "kvstore") as mock_kvstore:
                mock_kvstore.set = AsyncMock()
                mock_kvstore.get = AsyncMock()

                yield adapter, mock_conn, mock_cursor

    async def test_initialization(self, pgvector_adapter):
        """Test adapter initialization."""
        adapter, mock_conn, mock_cursor = pgvector_adapter

        # Mock the kvstore_impl function
        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.kvstore_impl") as mock_kvstore_impl:
            mock_kvstore = AsyncMock()
            mock_kvstore_impl.return_value = mock_kvstore

            # Mock the check_extension_version function response
            mock_cursor.fetchone.return_value = ["0.5.0"]

            await adapter.initialize()

            assert adapter.conn is not None
            mock_cursor.execute.assert_called()

    async def test_register_vector_db(self, pgvector_adapter):
        """Test registering a vector database."""
        adapter, mock_conn, mock_cursor = pgvector_adapter

        vector_db = VectorDB(
            identifier="test-db",
            embedding_model="test-model",
            embedding_dimension=384,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        # Setup mocks
        adapter.kvstore = AsyncMock()
        adapter.conn = mock_conn

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.upsert_models"):
            await adapter.register_vector_db(vector_db)

        # Verify the vector DB was cached
        assert "test-db" in adapter.cache

    async def test_insert_chunks(self, pgvector_adapter):
        """Test inserting chunks."""
        adapter, mock_conn, mock_cursor = pgvector_adapter

        # Setup
        adapter.conn = mock_conn
        chunks = create_sample_chunks()

        # Mock the cached index
        mock_index = AsyncMock()
        adapter.cache["test-db"] = mock_index

        await adapter.insert_chunks("test-db", chunks)

        # Verify insert_chunks was called on the index
        mock_index.insert_chunks.assert_called_once_with(chunks)

    async def test_delete_chunks(self, pgvector_adapter):
        """Test deleting specific chunks."""
        adapter, mock_conn, mock_cursor = pgvector_adapter

        # Setup
        adapter.conn = mock_conn

        # Mock the cached index
        mock_index = AsyncMock()
        mock_index.index = AsyncMock()
        adapter.cache["test-db"] = mock_index

        chunk_ids = ["chunk-1", "chunk-2"]
        await adapter.delete_chunks("test-db", chunk_ids)

        # Verify delete_chunk was called for each chunk
        assert mock_index.index.delete_chunk.call_count == len(chunk_ids)
