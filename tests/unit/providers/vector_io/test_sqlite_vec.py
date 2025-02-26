# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import sqlite3

import numpy as np
import pytest
import pytest_asyncio
import sqlite_vec

from llama_stack.apis.vector_io import Chunk, QueryChunksResponse
from llama_stack.providers.inline.vector_io.sqlite_vec.sqlite_vec import (
    SQLiteVecIndex,
    SQLiteVecVectorIOAdapter,
    generate_chunk_id,
)

# This test is a unit test for the SQLiteVecVectorIOAdapter class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_sqlite_vec.py \
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


@pytest_asyncio.fixture(scope="session", autouse=True)
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
async def test_query_chunks_vector(sqlite_vec_index, sample_chunks, sample_embeddings):
    await sqlite_vec_index.add_chunks(sample_chunks, sample_embeddings)

    query_embedding = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
    response = await sqlite_vec_index.query(
        query_embedding, query_str="", k=2, score_threshold=0.0, search_mode="vector"
    )
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2


import pytest


@pytest.mark.asyncio
async def test_query_chunks_full_text_search(sqlite_vec_index, sample_chunks, sample_embeddings):
    await sqlite_vec_index.add_chunks(sample_chunks, sample_embeddings)

    query_str = "Sentence 5"
    response = await sqlite_vec_index.query(
        embedding=None, k=3, score_threshold=0.0, query_str=query_str, search_mode="keyword"
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 3, f"Expected at least one result, but got {len(response.chunks)}"

    non_existent_query_str = "blablabla"
    response_no_results = await sqlite_vec_index.query(
        embedding=None, query_str=non_existent_query_str, k=1, score_threshold=0.0, search_mode="keyword"
    )

    assert isinstance(response_no_results, QueryChunksResponse)
    assert len(response_no_results.chunks) == 0, f"Expected 0 results, but got {len(response_no_results.chunks)}"


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
