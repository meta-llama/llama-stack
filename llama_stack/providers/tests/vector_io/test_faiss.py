# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import numpy as np
import pytest

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse
from llama_stack.providers.inline.vector_io.faiss.faiss import (
    FaissIndex,
    FaissVectorIOAdapter,
    FaissVectorIOConfig,
)
from llama_stack.providers.utils.kvstore.config import (
    SqliteKVStoreConfig,
)

# How to run this test:
#
# pytest llama_stack/providers/tests/vector_io/test_faiss.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto

FAISS_PROVIDER = "faiss"
EMBEDDING_DIMENSION = 384
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture(scope="session", autouse=True)
async def faiss_index():
    return await FaissIndex.create(dimension=EMBEDDING_DIMENSION, bank_id="test_bank")


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
async def test_add_chunks(faiss_index, sample_chunks, sample_embeddings):
    await faiss_index.add_chunks(sample_chunks, sample_embeddings)
    assert faiss_index.index.ntotal == len(sample_chunks)


@pytest.mark.asyncio
async def test_query_chunks(faiss_index, sample_chunks, sample_embeddings):
    await faiss_index.add_chunks(sample_chunks, sample_embeddings)
    query_embedding = np.random.rand(EMBEDDING_DIMENSION).astype(np.float32)
    response = await faiss_index.query(query_embedding, k=2, score_threshold=0.0)
    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 2


@pytest.fixture(scope="session")
async def faiss_adapter():
    # Create a mock config with kvstore
    kvstore_config = SqliteKVStoreConfig(db_path="./faiss_store.db")

    config = FaissVectorIOConfig(kvstore=kvstore_config)

    adapter = FaissVectorIOAdapter(config=config, inference_api=None)
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


@pytest.mark.asyncio
async def test_register_vector_db(faiss_adapter):
    vector_db = VectorDB(
        identifier="test_db",
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
        metadata={},
        provider_id=FAISS_PROVIDER,
    )
    await faiss_adapter.register_vector_db(vector_db)
    vector_dbs = await faiss_adapter.list_vector_dbs()
    assert any(db.identifier == "test_db" for db in vector_dbs)


@pytest.mark.asyncio
async def test_unregister_vector_db(faiss_adapter):
    vector_db = VectorDB(
        identifier="test_db",
        embedding_model=EMBEDDING_MODEL,
        embedding_dimension=EMBEDDING_DIMENSION,
        metadata={},
        provider_id=FAISS_PROVIDER,
    )
    await faiss_adapter.register_vector_db(vector_db)
    await faiss_adapter.unregister_vector_db("test_db")
    vector_dbs = await faiss_adapter.list_vector_dbs()
    assert not any(db.identifier == "test_db" for db in vector_dbs)
