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
from llama_stack.providers.utils.memory.vector_store import ChunkForDeletion

PGVECTOR_PROVIDER = "pgvector"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
def embedding_dimension():
    """Default embedding dimension for tests."""
    return 384


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
        # Use explicit COSINE distance metric for consistent testing
        index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="COSINE")

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
    def test_distance_metric_validation(self, embedding_dimension, mock_psycopg2_connection):
        connection, cursor = mock_psycopg2_connection

        vector_db = VectorDB(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
            index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="L2")
            assert index.distance_metric == "L2"
            with pytest.raises(ValueError, match="Distance metric 'INVALID' is not supported"):
                PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="INVALID")

    def test_get_pgvector_search_operator(self, pgvector_index):
        index, cursor = pgvector_index

        assert index.get_pgvector_search_operator() == "<=>"

        index.distance_metric = "L2"
        assert index.get_pgvector_search_operator() == "<->"

        index.distance_metric = "L1"
        assert index.get_pgvector_search_operator() == "<+>"

        index.distance_metric = "INNER_PRODUCT"
        assert index.get_pgvector_search_operator() == "<#>"

        index.distance_metric = "HAMMING"
        assert index.get_pgvector_search_operator() == "<~>"

        index.distance_metric = "JACCARD"
        assert index.get_pgvector_search_operator() == "<%>"

    def test_check_distance_metric_availability(self, pgvector_index):
        index, cursor = pgvector_index
        index.check_distance_metric_availability("COSINE")
        index.check_distance_metric_availability("L2")
        index.check_distance_metric_availability("L1")
        index.check_distance_metric_availability("INNER_PRODUCT")
        index.check_distance_metric_availability("HAMMING")
        index.check_distance_metric_availability("JACCARD")

        with pytest.raises(ValueError, match="Distance metric 'INVALID' is not supported"):
            index.check_distance_metric_availability("INVALID")

    async def test_add_chunks(self, pgvector_index, embedding_dimension):
        index, cursor = pgvector_index
        chunks = create_sample_chunks()
        embeddings = create_sample_embeddings(len(chunks), embedding_dimension)

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.execute_values") as mock_execute_values:
            await index.add_chunks(chunks, embeddings)

            assert mock_execute_values.called
            call_args = mock_execute_values.call_args

            query_arg = str(call_args[0][1])
            assert "INSERT INTO" in query_arg
            assert "content_tsvector" in query_arg

    async def test_query_vector(self, pgvector_index, embedding_dimension):
        index, cursor = pgvector_index
        query_embedding = create_sample_embeddings(1, embedding_dimension)[0]

        mock_results = [
            ({"content": "test content", "metadata": {}}, 0.1),
            ({"content": "test content 2", "metadata": {}}, 0.2),
        ]
        cursor.fetchall.return_value = mock_results

        response = await index.query_vector(query_embedding, k=2, score_threshold=0.0)

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) == 2
        assert len(response.scores) == 2

        call_args = cursor.execute.call_args
        assert "<=>" in str(call_args) or "ORDER BY" in str(call_args)

    async def test_query_vector_different_metrics(self, pgvector_index, embedding_dimension):
        index, cursor = pgvector_index
        query_embedding = create_sample_embeddings(1, embedding_dimension)[0]

        mock_results = [
            ({"content": "test content", "metadata": {}}, 0.1),
        ]
        cursor.fetchall.return_value = mock_results

        # Test L2 distance
        index.distance_metric = "L2"
        await index.query_vector(query_embedding, k=1, score_threshold=0.0)
        call_args = cursor.execute.call_args
        assert "<->" in str(call_args[0][0])  # L2 operator

        # Test L1 distance
        index.distance_metric = "L1"
        await index.query_vector(query_embedding, k=1, score_threshold=0.0)
        call_args = cursor.execute.call_args
        assert "<+>" in str(call_args[0][0])  # L1 operator

        # Test INNER_PRODUCT distance
        index.distance_metric = "INNER_PRODUCT"
        await index.query_vector(query_embedding, k=1, score_threshold=0.0)
        call_args = cursor.execute.call_args
        assert "<#>" in str(call_args[0][0])  # Inner product operator

        # Test Hamming distance
        index.distance_metric = "HAMMING"
        await index.query_vector(query_embedding, k=1, score_threshold=0.0)
        call_args = cursor.execute.call_args
        assert "<~>" in str(call_args[0][0])  # Hamming operator

        # Test Jaccard distance
        index.distance_metric = "JACCARD"
        await index.query_vector(query_embedding, k=1, score_threshold=0.0)
        call_args = cursor.execute.call_args
        assert "<%>" in str(call_args[0][0])  # Jaccard operator

    async def test_query_keyword(self, pgvector_index):
        index, cursor = pgvector_index
        query_string = "machine learning"

        mock_results = [
            ({"content": "Machine learning is great", "metadata": {}}, 0.8),
            ({"content": "Learning machines are useful", "metadata": {}}, 0.6),
        ]
        cursor.fetchall.return_value = mock_results

        response = await index.query_keyword(query_string, k=2, score_threshold=0.0)

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) == 2
        assert len(response.scores) == 2

        call_args = cursor.execute.call_args
        assert "ts_rank" in str(call_args) or "plainto_tsquery" in str(call_args)

    async def test_query_keyword_with_score_threshold(self, pgvector_index):
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

        assert len(response.chunks) == 1
        assert response.scores[0] >= score_threshold

    async def test_query_hybrid_rrf(self, pgvector_index, embedding_dimension):
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
        assert cursor.execute.call_count >= 2

    async def test_query_hybrid_weighted(self, pgvector_index, embedding_dimension):
        index, cursor = pgvector_index
        query_embedding = create_sample_embeddings(1, embedding_dimension)[0]
        query_string = "machine learning"

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

    def test_constructor_invalid_distance_metric(self, embedding_dimension, mock_psycopg2_connection):
        connection, cursor = mock_psycopg2_connection

        vector_db = VectorDB(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
            with pytest.raises(ValueError, match="Distance metric 'INVALID_METRIC' is not supported by PGVector"):
                PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="INVALID_METRIC")

            with pytest.raises(ValueError, match="Supported metrics are:"):
                PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="UNKNOWN")

            try:
                index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric="COSINE")
                assert index.distance_metric == "COSINE"
            except ValueError:
                pytest.fail("Valid distance metric 'COSINE' should not raise ValueError")

    def test_constructor_all_supported_distance_metrics(self, embedding_dimension, mock_psycopg2_connection):
        connection, cursor = mock_psycopg2_connection

        vector_db = VectorDB(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        supported_metrics = ["L2", "L1", "COSINE", "INNER_PRODUCT", "HAMMING", "JACCARD"]

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
            for metric in supported_metrics:
                try:
                    index = PGVectorIndex(vector_db, embedding_dimension, connection, distance_metric=metric)
                    assert index.distance_metric == metric

                    expected_operators = {
                        "L2": "<->",
                        "L1": "<+>",
                        "COSINE": "<=>",
                        "INNER_PRODUCT": "<#>",
                        "HAMMING": "<~>",
                        "JACCARD": "<%>",
                    }
                    assert index.get_pgvector_search_operator() == expected_operators[metric]
                except Exception as e:
                    pytest.fail(f"Valid distance metric '{metric}' should not raise exception: {e}")

    def test_error_handling_in_constructor(self, embedding_dimension):
        vector_db = VectorDB(
            identifier="test-vector-db",
            embedding_model="test-model",
            embedding_dimension=embedding_dimension,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.psycopg2"):
            mock_connection = MagicMock()

            mock_cursor_context = MagicMock()
            mock_cursor_context.__enter__.side_effect = Exception("Database connection failed")
            mock_cursor_context.__exit__ = MagicMock()

            mock_connection.cursor.return_value = mock_cursor_context

            with pytest.raises(RuntimeError, match="Error creating PGVectorIndex"):
                PGVectorIndex(vector_db, embedding_dimension, mock_connection, distance_metric="COSINE")

    async def test_delete_chunks(self, pgvector_index):
        index, cursor = pgvector_index

        chunks_for_deletion = [
            ChunkForDeletion(chunk_id="test-chunk-1", document_id="doc-1"),
            ChunkForDeletion(chunk_id="test-chunk-2", document_id="doc-2"),
        ]

        await index.delete_chunks(chunks_for_deletion)

        cursor.execute.assert_called()
        call_args = cursor.execute.call_args
        assert "DELETE FROM" in str(call_args)
        assert "test-chunk-1" in str(call_args) or "test-chunk-2" in str(call_args)

    async def test_delete_index(self, pgvector_index):
        """Test deleting the entire index."""
        index, cursor = pgvector_index

        await index.delete()

        cursor.execute.assert_called_with(f"DROP TABLE IF EXISTS {index.table_name}")


class TestPGVectorVectorIOAdapter:
    @pytest.fixture
    async def pgvector_adapter(self):
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
        adapter, mock_conn, mock_cursor = pgvector_adapter

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.kvstore_impl") as mock_kvstore_impl:
            mock_kvstore = AsyncMock()
            mock_kvstore_impl.return_value = mock_kvstore

            mock_cursor.fetchone.return_value = ["0.5.0"]

            await adapter.initialize()

            assert adapter.conn is not None
            mock_cursor.execute.assert_called()

            assert adapter.metadata_collection_name == "openai_vector_stores_metadata"

    async def test_register_vector_db(self, pgvector_adapter):
        adapter, mock_conn, mock_cursor = pgvector_adapter

        vector_db = VectorDB(
            identifier="test-db",
            embedding_model="test-model",
            embedding_dimension=384,
            provider_id=PGVECTOR_PROVIDER,
            provider_resource_id=f"{PGVECTOR_PROVIDER}:test-vector-db",
        )

        adapter.kvstore = AsyncMock()
        adapter.conn = mock_conn

        with patch("llama_stack.providers.remote.vector_io.pgvector.pgvector.upsert_models"):
            await adapter.register_vector_db(vector_db)

        assert "test-db" in adapter.cache

    async def test_insert_chunks(self, pgvector_adapter):
        adapter, mock_conn, mock_cursor = pgvector_adapter

        adapter.conn = mock_conn
        chunks = create_sample_chunks()

        mock_index = AsyncMock()
        adapter.cache["test-db"] = mock_index

        await adapter.insert_chunks("test-db", chunks)
        mock_index.insert_chunks.assert_called_once_with(chunks)

    async def test_delete_chunks(self, pgvector_adapter):
        adapter, mock_conn, mock_cursor = pgvector_adapter
        adapter.conn = mock_conn
        mock_index = AsyncMock()
        mock_index.index = AsyncMock()
        adapter.cache["test-db"] = mock_index

        chunks_for_deletion = [
            ChunkForDeletion(chunk_id="chunk-1", document_id="doc-1"),
            ChunkForDeletion(chunk_id="chunk-2", document_id="doc-2"),
        ]
        await adapter.delete_chunks("test-db", chunks_for_deletion)

        mock_index.index.delete_chunks.assert_called_once_with(chunks_for_deletion)
