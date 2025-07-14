# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import random

import numpy as np
import pytest
from pymilvus import MilvusClient, connections

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, ChunkMetadata
from llama_stack.providers.inline.vector_io.faiss.config import FaissVectorIOConfig
from llama_stack.providers.inline.vector_io.faiss.faiss import FaissIndex, FaissVectorIOAdapter
from llama_stack.providers.inline.vector_io.milvus.config import MilvusVectorIOConfig, SqliteKVStoreConfig
from llama_stack.providers.inline.vector_io.sqlite_vec import SQLiteVectorIOConfig
from llama_stack.providers.inline.vector_io.sqlite_vec.sqlite_vec import SQLiteVecIndex, SQLiteVecVectorIOAdapter
from llama_stack.providers.remote.vector_io.milvus.milvus import MilvusIndex, MilvusVectorIOAdapter

EMBEDDING_DIMENSION = 384
COLLECTION_PREFIX = "test_collection"
MILVUS_ALIAS = "test_milvus"


@pytest.fixture
def vector_db_id() -> str:
    return f"test-vector-db-{random.randint(1, 100)}"


@pytest.fixture(scope="session")
def embedding_dimension() -> int:
    return EMBEDDING_DIMENSION


@pytest.fixture(scope="session")
def sample_chunks():
    """Generates chunks that force multiple batches for a single document to expose ID conflicts."""
    n, k = 10, 3
    sample = [
        Chunk(content=f"Sentence {i} from document {j}", metadata={"document_id": f"document-{j}"})
        for j in range(k)
        for i in range(n)
    ]
    sample.extend(
        [
            Chunk(
                content=f"Sentence {i} from document {j + k}",
                chunk_metadata=ChunkMetadata(
                    document_id=f"document-{j + k}",
                    chunk_id=f"document-{j}-chunk-{i}",
                    source=f"example source-{j + k}-{i}",
                ),
            )
            for j in range(k)
            for i in range(n)
        ]
    )
    return sample


@pytest.fixture(scope="session")
def sample_chunks_with_metadata():
    """Generates chunks that force multiple batches for a single document to expose ID conflicts."""
    n, k = 10, 3
    sample = [
        Chunk(
            content=f"Sentence {i} from document {j}",
            metadata={"document_id": f"document-{j}"},
            chunk_metadata=ChunkMetadata(
                document_id=f"document-{j}",
                chunk_id=f"document-{j}-chunk-{i}",
                source=f"example source-{j}-{i}",
            ),
        )
        for j in range(k)
        for i in range(n)
    ]
    return sample


@pytest.fixture(scope="session")
def sample_embeddings(sample_chunks):
    np.random.seed(42)
    return np.array([np.random.rand(EMBEDDING_DIMENSION).astype(np.float32) for _ in sample_chunks])


@pytest.fixture(scope="session")
def sample_embeddings_with_metadata(sample_chunks_with_metadata):
    np.random.seed(42)
    return np.array([np.random.rand(EMBEDDING_DIMENSION).astype(np.float32) for _ in sample_chunks_with_metadata])


@pytest.fixture(params=["milvus", "sqlite_vec", "faiss"])
def vector_provider(request):
    return request.param


@pytest.fixture(scope="session")
def mock_inference_api(embedding_dimension):
    class MockInferenceAPI:
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [np.random.rand(embedding_dimension).astype(np.float32).tolist() for _ in texts]

    return MockInferenceAPI()


@pytest.fixture
async def unique_kvstore_config(tmp_path_factory):
    # Generate a unique filename for this test
    unique_id = f"test_kv_{np.random.randint(1e6)}"
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / f"{unique_id}.db")

    return SqliteKVStoreConfig(db_path=db_path)


@pytest.fixture(scope="session")
def sqlite_vec_db_path(tmp_path_factory):
    db_path = str(tmp_path_factory.getbasetemp() / "test_sqlite_vec.db")
    return db_path


@pytest.fixture
async def sqlite_vec_vec_index(embedding_dimension, tmp_path_factory):
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / f"test_sqlite_vec_{np.random.randint(1e6)}.db")
    bank_id = f"sqlite_vec_bank_{np.random.randint(1e6)}"
    index = SQLiteVecIndex(embedding_dimension, db_path, bank_id)
    await index.initialize()
    index.db_path = db_path
    yield index
    index.delete()


@pytest.fixture
async def sqlite_vec_adapter(sqlite_vec_db_path, mock_inference_api, embedding_dimension):
    config = SQLiteVectorIOConfig(
        db_path=sqlite_vec_db_path,
        kvstore=SqliteKVStoreConfig(),
    )
    adapter = SQLiteVecVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    collection_id = f"sqlite_test_collection_{np.random.randint(1e6)}"
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=collection_id,
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=embedding_dimension,
        )
    )
    adapter.test_collection_id = collection_id
    yield adapter
    await adapter.shutdown()


@pytest.fixture(scope="session")
def milvus_vec_db_path(tmp_path_factory):
    db_path = str(tmp_path_factory.getbasetemp() / "test_milvus.db")
    return db_path


@pytest.fixture
async def milvus_vec_index(milvus_vec_db_path, embedding_dimension):
    client = MilvusClient(milvus_vec_db_path)
    name = f"{COLLECTION_PREFIX}_{np.random.randint(1e6)}"
    connections.connect(alias=MILVUS_ALIAS, uri=milvus_vec_db_path)
    index = MilvusIndex(client, name, consistency_level="Strong")
    index.db_path = milvus_vec_db_path
    yield index


@pytest.fixture
async def milvus_vec_adapter(milvus_vec_db_path, mock_inference_api):
    config = MilvusVectorIOConfig(
        db_path=milvus_vec_db_path,
        kvstore=SqliteKVStoreConfig(),
    )
    adapter = MilvusVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=adapter.metadata_collection_name,
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=128,
        )
    )
    yield adapter
    await adapter.shutdown()


@pytest.fixture
def faiss_vec_db_path(tmp_path_factory):
    db_path = str(tmp_path_factory.getbasetemp() / "test_faiss.db")
    return db_path


@pytest.fixture
async def faiss_vec_index(embedding_dimension):
    index = FaissIndex(embedding_dimension)
    yield index
    await index.delete()


@pytest.fixture
async def faiss_vec_adapter(unique_kvstore_config, mock_inference_api, embedding_dimension):
    config = FaissVectorIOConfig(
        kvstore=unique_kvstore_config,
    )
    adapter = FaissVectorIOAdapter(
        config=config,
        inference_api=mock_inference_api,
        files_api=None,
    )
    await adapter.initialize()
    await adapter.register_vector_db(
        VectorDB(
            identifier=f"faiss_test_collection_{np.random.randint(1e6)}",
            provider_id="test_provider",
            embedding_model="test_model",
            embedding_dimension=embedding_dimension,
        )
    )
    yield adapter
    await adapter.shutdown()


@pytest.fixture
def vector_io_adapter(vector_provider, request):
    """Returns the appropriate vector IO adapter based on the provider parameter."""
    if vector_provider == "milvus":
        return request.getfixturevalue("milvus_vec_adapter")
    elif vector_provider == "faiss":
        return request.getfixturevalue("faiss_vec_adapter")
    else:
        return request.getfixturevalue("sqlite_vec_adapter")


@pytest.fixture
def vector_index(vector_provider, request):
    """Returns appropriate vector index based on provider parameter"""
    return request.getfixturevalue(f"{vector_provider}_vec_index")
