# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import sqlite3

import numpy as np
import pytest
import sqlite_vec

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse
from llama_stack.providers.inline.vector_io.sqlite_vec.sqlite_vec import SQLiteVecIndex, SQLiteVecVectorIOAdapter

# How to run this test:
#
# pytest llama_stack/providers/tests/vector_io/test_sqlite_vec.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


@pytest.fixture(scope="session")
def loop():
    return asyncio.get_event_loop()


@pytest.fixture(scope="session", autouse=True)
def sqlite_connection(loop):
    conn = sqlite3.connect(":memory:")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    yield conn
    conn.close()


@pytest.fixture(scope="session", autouse=True)
async def sqlite_vec_index(sqlite_connection):
    return await SQLiteVecIndex.create(dimension=384, connection=sqlite_connection, bank_id="test_bank")


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            content="Python is a high-level programming language.",
            metadata={"category": "programming", "document_id": "doc 1"},
        ),
        Chunk(
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "AI", "document_id": "doc 1"},
        ),
    ]


@pytest.fixture
def sample_embeddings():
    np.random.seed(42)
    return np.array(
        [
            np.random.rand(384).astype(np.float32),
            np.random.rand(384).astype(np.float32),
        ]
    )


@pytest.mark.asyncio
async def test_add_chunks(sqlite_vec_index, sample_chunks, sample_embeddings):
    await sqlite_vec_index.add_chunks(sample_chunks, sample_embeddings)
    cur = sqlite_vec_index.connection.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {sqlite_vec_index.metadata_table}")
    count = cur.fetchone()[0]
    assert count == len(sample_chunks)


@pytest.mark.asyncio
async def test_query_chunks(sqlite_vec_index, sample_chunks, sample_embeddings):
    await sqlite_vec_index.add_chunks(sample_chunks, sample_embeddings)
    query_embedding = np.random.rand(384).astype(np.float32)
    response = await sqlite_vec_index.query(query_embedding, k=1, score_threshold=0.0)
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) > 0


@pytest.fixture
async def sqlite_vec_adapter(sqlite_connection):
    config = type("Config", (object,), {"db_path": ":memory:"})  # Mock config with in-memory database
    adapter = SQLiteVecVectorIOAdapter(config=config, inference_api=None)
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


@pytest.mark.asyncio
async def test_register_vector_db(sqlite_vec_adapter):
    vector_db = VectorDB(
        identifier="test_db",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        metadata={},
        provider_id="sqlite_vec",
    )
    await sqlite_vec_adapter.register_vector_db(vector_db)
    vector_dbs = await sqlite_vec_adapter.list_vector_dbs()
    assert any(db.identifier == "test_db" for db in vector_dbs)


@pytest.mark.asyncio
async def test_unregister_vector_db(sqlite_vec_adapter):
    vector_db = VectorDB(
        identifier="test_db",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        metadata={},
        provider_id="sqlite_vec",
    )
    await sqlite_vec_adapter.register_vector_db(vector_db)
    await sqlite_vec_adapter.unregister_vector_db("test_db")
    vector_dbs = await sqlite_vec_adapter.list_vector_dbs()
    assert not any(db.identifier == "test_db" for db in vector_dbs)
