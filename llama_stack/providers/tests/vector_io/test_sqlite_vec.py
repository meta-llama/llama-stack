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
from llama_stack.providers.inline.vector_io.sqlite_vec.sqlite_vec import (
    SQLiteVecIndex,
    SQLiteVecVectorIOAdapter,
    generate_chunk_id,
)

# How to run this test:
#
# pytest llama_stack/providers/tests/vector_io/test_sqlite_vec.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

SQLITE_VEC_PROVIDER = "sqlite_vec"
EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture(scope="session", autouse=True)
def sqlite_connection(loop):
    conn = sqlite3.connect(":memory:")
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="session", autouse=True)
async def sqlite_vec_index(sqlite_connection):
    return await SQLiteVecIndex.create(dimension=EMBEDDING_DIMENSION, connection=sqlite_connection, bank_id="test_bank")


@pytest.fixture(scope="session")
def sample_chunks():
    """Generates chunks that force multiple batches for a single document to expose ID conflicts."""
    n, k = 10, 3
    sample = [
        Chunk(content=f"Sentence {i} from document {j}", metadata={"document_id": f"document-{j}"})
        for j in range(k)
        for i in range(n)
    ]
    return sample


@pytest.fixture(scope="session")
def sample_embeddings(sample_chunks):
    np.random.seed(42)
    return np.array([np.random.rand(EMBEDDING_DIMENSION).astype(np.float32) for _ in sample_chunks])


@pytest.mark.asyncio
async def test_add_chunks(sqlite_vec_index, sample_chunks, sample_embeddings):
    await sqlite_vec_index.add_chunks(sample_chunks, sample_embeddings, batch_size=2)
    cur = sqlite_vec_index.connection.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {sqlite_vec_index.metadata_table}")
    count = cur.fetchone()[0]
    assert count == len(sample_chunks)


@pytest.mark.asyncio
async def test_query_chunks(sqlite_vec_index, sample_chunks, sample_embeddings):
    await sqlite_vec_index.add_chunks(sample_chunks, sample_embeddings)
    query_embedding = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
    response = await sqlite_vec_index.query(query_embedding, k=2, score_threshold=0.0)
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2


@pytest.mark.asyncio
async def test_chunk_id_conflict(sqlite_vec_index, sample_chunks):
    """Test that chunk IDs do not conflict across batches when inserting chunks."""
    # Reduce batch size to force multiple batches for same document
    # since there are 10 chunks per document and batch size is 2
    batch_size = 2
    sample_embeddings = np.random.rand(len(sample_chunks), EMBEDDING_DIMENSION).astype(np.float32)

    await sqlite_vec_index.add_chunks(sample_chunks, sample_embeddings, batch_size=batch_size)

    cur = sqlite_vec_index.connection.cursor()

    # Retrieve all chunk IDs to check for duplicates
    cur.execute(f"SELECT id FROM {sqlite_vec_index.metadata_table}")
    chunk_ids = [row[0] for row in cur.fetchall()]
    cur.close()

    # Ensure all chunk IDs are unique
    assert len(chunk_ids) == len(set(chunk_ids)), "Duplicate chunk IDs detected across batches!"


@pytest.fixture(scope="session")
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
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
        metadata={},
        provider_id=SQLITE_VEC_PROVIDER,
    )
    await sqlite_vec_adapter.register_vector_db(vector_db)
    vector_dbs = await sqlite_vec_adapter.list_vector_dbs()
    assert any(db.identifier == "test_db" for db in vector_dbs)


@pytest.mark.asyncio
async def test_unregister_vector_db(sqlite_vec_adapter):
    vector_db = VectorDB(
        identifier="test_db",
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
        metadata={},
        provider_id=SQLITE_VEC_PROVIDER,
    )
    await sqlite_vec_adapter.register_vector_db(vector_db)
    await sqlite_vec_adapter.unregister_vector_db("test_db")
    vector_dbs = await sqlite_vec_adapter.list_vector_dbs()
    assert not any(db.identifier == "test_db" for db in vector_dbs)


def test_generate_chunk_id():
    chunks = [
        Chunk(content="test", metadata={"document_id": "doc-1"}),
        Chunk(content="test ", metadata={"document_id": "doc-1"}),
        Chunk(content="test 3", metadata={"document_id": "doc-1"}),
    ]

    chunk_ids = sorted([generate_chunk_id(chunk.metadata["document_id"], chunk.content) for chunk in chunks])
    assert chunk_ids == [
        "177a1368-f6a8-0c50-6e92-18677f2c3de3",
        "bc744db3-1b25-0a9c-cdff-b6ba3df73c36",
        "f68df25d-d9aa-ab4d-5684-64a233add20d",
    ]
