# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import random
from unittest.mock import AsyncMock

import numpy as np
import pytest

from llama_stack.apis.inference import EmbeddingsResponse, Inference
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse
from llama_stack.providers.remote.vector_io.opengauss.config import (
    OpenGaussVectorIOConfig,
)
from llama_stack.providers.remote.vector_io.opengauss.opengauss import (
    OpenGaussIndex,
    OpenGaussVectorIOAdapter,
)
from llama_stack.providers.utils.kvstore.config import (
    SqliteKVStoreConfig,
)

# Skip all tests in this file if the required environment variables are not set.
pytestmark = pytest.mark.skipif(
    not all(
        os.getenv(var)
        for var in [
            "OPENGAUSS_HOST",
            "OPENGAUSS_PORT",
            "OPENGAUSS_DB",
            "OPENGAUSS_USER",
            "OPENGAUSS_PASSWORD",
        ]
    ),
    reason="OpenGauss connection environment variables not set",
)


@pytest.fixture(scope="session")
def embedding_dimension() -> int:
    return 128


@pytest.fixture
def sample_chunks():
    """Provides a list of sample chunks for testing."""
    return [
        Chunk(
            content="The sky is blue.",
            metadata={"document_id": "doc1", "topic": "nature"},
        ),
        Chunk(
            content="An apple a day keeps the doctor away.",
            metadata={"document_id": "doc2", "topic": "health"},
        ),
        Chunk(
            content="Quantum computing is a new frontier.",
            metadata={"document_id": "doc3", "topic": "technology"},
        ),
    ]


@pytest.fixture
def sample_embeddings(embedding_dimension, sample_chunks):
    """Provides a deterministic set of embeddings for the sample chunks."""
    # Use a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    return rng.random((len(sample_chunks), embedding_dimension), dtype=np.float32)


@pytest.fixture
def mock_inference_api(sample_embeddings):
    """Mocks the inference API to return dummy embeddings."""
    mock_api = AsyncMock(spec=Inference)
    mock_api.embeddings = AsyncMock(return_value=EmbeddingsResponse(embeddings=sample_embeddings.tolist()))
    return mock_api


@pytest.fixture
def vector_db(embedding_dimension):
    """Provides a sample VectorDB object for registration."""
    return VectorDB(
        identifier=f"test_db_{random.randint(1, 10000)}",
        embedding_model="test_embedding_model",
        embedding_dimension=embedding_dimension,
        provider_id="opengauss",
    )


@pytest.fixture
async def opengauss_connection():
    """Creates and manages a connection to the OpenGauss database."""
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("OPENGAUSS_HOST"),
        port=int(os.getenv("OPENGAUSS_PORT")),
        database=os.getenv("OPENGAUSS_DB"),
        user=os.getenv("OPENGAUSS_USER"),
        password=os.getenv("OPENGAUSS_PASSWORD"),
    )
    conn.autocommit = True
    yield conn
    conn.close()


@pytest.fixture
async def opengauss_index(opengauss_connection, vector_db):
    """Fixture to create and clean up an OpenGaussIndex instance."""
    index = OpenGaussIndex(vector_db, vector_db.embedding_dimension, opengauss_connection)
    yield index
    await index.delete()


@pytest.fixture
async def opengauss_adapter(mock_inference_api):
    """Fixture to set up and tear down the OpenGaussVectorIOAdapter."""
    config = OpenGaussVectorIOConfig(
        host=os.getenv("OPENGAUSS_HOST"),
        port=int(os.getenv("OPENGAUSS_PORT")),
        db=os.getenv("OPENGAUSS_DB"),
        user=os.getenv("OPENGAUSS_USER"),
        password=os.getenv("OPENGAUSS_PASSWORD"),
        kvstore=SqliteKVStoreConfig(db_name="opengauss_test.db"),
    )
    adapter = OpenGaussVectorIOAdapter(config, mock_inference_api)
    await adapter.initialize()
    yield adapter
    if adapter.conn and not adapter.conn.closed:
        for db_id in list(adapter.cache.keys()):
            try:
                await adapter.unregister_vector_db(db_id)
            except Exception as e:
                print(f"Error during cleanup of {db_id}: {e}")
    await adapter.shutdown()
    # Clean up the sqlite db file
    if os.path.exists("opengauss_test.db"):
        os.remove("opengauss_test.db")


class TestOpenGaussIndex:
    async def test_add_and_query_vector(self, opengauss_index, sample_chunks, sample_embeddings):
        """Test adding chunks with embeddings and querying for the most similar one."""
        await opengauss_index.add_chunks(sample_chunks, sample_embeddings)

        # Query with the embedding of the first chunk
        query_embedding = sample_embeddings[0]
        response = await opengauss_index.query_vector(query_embedding, k=1, score_threshold=0.0)

        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) == 1
        assert response.chunks[0].content == sample_chunks[0].content
        # The distance to itself should be 0, resulting in infinite score
        assert response.scores[0] == float("inf")


class TestOpenGaussVectorIOAdapter:
    async def test_initialization(self, opengauss_adapter):
        """Test that the adapter initializes and connects to the database."""
        assert opengauss_adapter.conn is not None
        assert not opengauss_adapter.conn.closed

    async def test_register_and_unregister_vector_db(self, opengauss_adapter, vector_db):
        """Test the registration and unregistration of a vector database."""
        await opengauss_adapter.register_vector_db(vector_db)
        assert vector_db.identifier in opengauss_adapter.cache

        table_name = opengauss_adapter.cache[vector_db.identifier].index.table_name
        with opengauss_adapter.conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = %s);",
                (table_name,),
            )
            assert cur.fetchone()[0]

        await opengauss_adapter.unregister_vector_db(vector_db.identifier)
        assert vector_db.identifier not in opengauss_adapter.cache

        with opengauss_adapter.conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = %s);",
                (table_name,),
            )
            assert not cur.fetchone()[0]

    async def test_adapter_end_to_end_query(self, opengauss_adapter, vector_db, sample_chunks):
        """
        Tests the full adapter flow: text query -> embedding generation -> vector search.
        """
        # 1. Register the DB and insert chunks. The adapter will use the mocked
        #    inference_api to generate embeddings for these chunks.
        await opengauss_adapter.register_vector_db(vector_db)
        await opengauss_adapter.insert_chunks(vector_db.identifier, sample_chunks)

        # 2. The user query is a text string.
        query_text = "What is the color of the sky?"

        # 3. The adapter will now internally call the (mocked) inference_api
        #    to get an embedding for the query_text.
        response = await opengauss_adapter.query_chunks(vector_db.identifier, query_text)

        # 4. Assertions
        assert isinstance(response, QueryChunksResponse)
        assert len(response.chunks) > 0

        # Because the mocked inference_api returns random embeddings, we can't
        # deterministically know which chunk is "closest". However, in a real
        # integration test with a real model, this assertion would be more specific.
        # For this unit test, we just confirm that the process completes and returns data.
        assert response.chunks[0].content in [c.content for c in sample_chunks]
