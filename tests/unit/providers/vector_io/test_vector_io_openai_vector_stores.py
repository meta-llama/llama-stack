# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import time
from unittest.mock import AsyncMock

import numpy as np
import pytest
import pytest_asyncio
from pymilvus import Collection, MilvusClient, connections

from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, QueryChunksResponse
from llama_stack.providers.inline.vector_io.milvus.config import MilvusVectorIOConfig, SqliteKVStoreConfig
from llama_stack.providers.remote.vector_io.milvus.milvus import VECTOR_DBS_PREFIX, MilvusIndex, MilvusVectorIOAdapter
from llama_stack.providers.utils.kvstore import kvstore_impl

# TODO: Refactor these to be for inline vector-io providers
MILVUS_ALIAS = "test_milvus"
COLLECTION_PREFIX = "test_collection"


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture(scope="session")
def mock_inference_api(embedding_dimension):
    class MockInferenceAPI:
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [np.random.rand(embedding_dimension).astype(np.float32).tolist() for _ in texts]

    return MockInferenceAPI()


@pytest_asyncio.fixture
async def unique_kvstore_config(tmp_path_factory):
    # Generate a unique filename for this test
    unique_id = f"test_kv_{np.random.randint(1e6)}"
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / f"{unique_id}.db")

    return SqliteKVStoreConfig(db_path=db_path)


@pytest_asyncio.fixture(scope="session", autouse=True)
async def milvus_vec_index(embedding_dimension, tmp_path_factory):
    temp_dir = tmp_path_factory.getbasetemp()
    db_path = str(temp_dir / "test_milvus.db")
    client = MilvusClient(db_path)
    name = f"{COLLECTION_PREFIX}_{np.random.randint(1e6)}"
    connections.connect(alias=MILVUS_ALIAS, uri=db_path)
    index = MilvusIndex(client, name, consistency_level="Strong")
    index.db_path = db_path
    yield index


@pytest_asyncio.fixture(scope="session")
async def milvus_vec_adapter(milvus_vec_index, mock_inference_api):
    config = MilvusVectorIOConfig(
        db_path=milvus_vec_index.db_path,
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


@pytest.mark.asyncio
async def test_cache_contains_initial_collection(milvus_vec_adapter):
    coll_name = milvus_vec_adapter.metadata_collection_name
    assert coll_name in milvus_vec_adapter.cache


@pytest.mark.asyncio
async def test_add_chunks(milvus_vec_index, sample_chunks, sample_embeddings):
    await milvus_vec_index.add_chunks(sample_chunks, sample_embeddings)
    resp = await milvus_vec_index.query_vector(sample_embeddings[0], k=1, score_threshold=-1)
    assert resp.chunks[0].content == sample_chunks[0].content


@pytest.mark.asyncio
async def test_query_chunks_vector(milvus_vec_index, sample_chunks, sample_embeddings, embedding_dimension):
    await milvus_vec_index.add_chunks(sample_chunks, sample_embeddings)
    query_emb = np.random.rand(embedding_dimension).astype(np.float32)
    resp = await milvus_vec_index.query_vector(query_emb, k=2, score_threshold=0.0)
    assert isinstance(resp, QueryChunksResponse)
    assert len(resp.chunks) == 2


@pytest.mark.asyncio
async def test_chunk_id_conflict(milvus_vec_index, sample_chunks, embedding_dimension):
    embeddings = np.random.rand(len(sample_chunks), embedding_dimension).astype(np.float32)
    await milvus_vec_index.add_chunks(sample_chunks, embeddings)
    coll = Collection(milvus_vec_index.collection_name, using=MILVUS_ALIAS)
    ids = coll.query(expr="id >= 0", output_fields=["id"], timeout=30)
    flat_ids = [i["id"] for i in ids]
    assert len(flat_ids) == len(set(flat_ids))


@pytest.mark.asyncio
async def test_initialize_with_milvus_client(milvus_vec_index, unique_kvstore_config):
    kvstore = await kvstore_impl(unique_kvstore_config)
    vector_db = VectorDB(
        identifier="test_db",
        provider_id="test_provider",
        embedding_model="test_model",
        embedding_dimension=128,
        metadata={"test_key": "test_value"},
    )
    test_vector_db_data = vector_db.model_dump_json()
    await kvstore.set(f"{VECTOR_DBS_PREFIX}test_db", test_vector_db_data)
    tmp_milvus_vec_adapter = MilvusVectorIOAdapter(
        config=MilvusVectorIOConfig(
            db_path=milvus_vec_index.db_path,
            kvstore=unique_kvstore_config,
        ),
        inference_api=None,
        files_api=None,
    )
    await tmp_milvus_vec_adapter.initialize()

    vector_db = VectorDB(
        identifier="test_db",
        provider_id="test_provider",
        embedding_model="test_model",
        embedding_dimension=128,
    )
    test_vector_db_data = vector_db.model_dump_json()
    await tmp_milvus_vec_adapter.kvstore.set(f"{VECTOR_DBS_PREFIX}/test_db", test_vector_db_data)

    assert milvus_vec_index.client is not None
    assert isinstance(milvus_vec_index.client, MilvusClient)
    assert tmp_milvus_vec_adapter.cache is not None
    # registering a vector won't update the cache or openai_vector_store collection name
    assert (
        tmp_milvus_vec_adapter.metadata_collection_name not in tmp_milvus_vec_adapter.cache
        or tmp_milvus_vec_adapter.openai_vector_stores
    )


@pytest.mark.asyncio
async def test_persistence_across_adapter_restarts(
    tmp_path, milvus_vec_index, mock_inference_api, unique_kvstore_config
):
    adapter1 = MilvusVectorIOAdapter(
        config=MilvusVectorIOConfig(db_path=milvus_vec_index.db_path, kvstore=unique_kvstore_config),
        inference_api=mock_inference_api,
        files_api=None,
    )
    await adapter1.initialize()
    dummy = VectorDB(
        identifier="foo_db", provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )
    await adapter1.register_vector_db(dummy)
    await adapter1.shutdown()

    await adapter1.initialize()
    assert "foo_db" in adapter1.cache
    await adapter1.shutdown()


@pytest.mark.asyncio
async def test_register_and_unregister_vector_db(milvus_vec_adapter):
    try:
        connections.disconnect(MILVUS_ALIAS)
    except Exception as _:
        pass

    connections.connect(alias=MILVUS_ALIAS, uri=milvus_vec_adapter.config.db_path)
    unique_id = f"foo_db_{np.random.randint(1e6)}"
    dummy = VectorDB(
        identifier=unique_id, provider_id="test_provider", embedding_model="test_model", embedding_dimension=128
    )

    await milvus_vec_adapter.register_vector_db(dummy)
    assert dummy.identifier in milvus_vec_adapter.cache

    if dummy.identifier in milvus_vec_adapter.cache:
        index = milvus_vec_adapter.cache[dummy.identifier].index
        if hasattr(index, "client") and hasattr(index.client, "_using"):
            index.client._using = MILVUS_ALIAS

    await milvus_vec_adapter.unregister_vector_db(dummy.identifier)
    assert dummy.identifier not in milvus_vec_adapter.cache


@pytest.mark.asyncio
async def test_query_unregistered_raises(milvus_vec_adapter):
    fake_emb = np.zeros(8, dtype=np.float32)
    with pytest.raises(AttributeError):
        await milvus_vec_adapter.query_chunks("no_such_db", fake_emb)


@pytest.mark.asyncio
async def test_insert_chunks_calls_underlying_index(milvus_vec_adapter):
    fake_index = AsyncMock()
    milvus_vec_adapter._get_and_cache_vector_db_index = AsyncMock(return_value=fake_index)

    chunks = ["chunk1", "chunk2"]
    await milvus_vec_adapter.insert_chunks("db1", chunks)

    fake_index.insert_chunks.assert_awaited_once_with(chunks)


@pytest.mark.asyncio
async def test_insert_chunks_missing_db_raises(milvus_vec_adapter):
    milvus_vec_adapter._get_and_cache_vector_db_index = AsyncMock(return_value=None)

    with pytest.raises(ValueError):
        await milvus_vec_adapter.insert_chunks("db_not_exist", [])


@pytest.mark.asyncio
async def test_query_chunks_calls_underlying_index_and_returns(milvus_vec_adapter):
    expected = QueryChunksResponse(chunks=[Chunk(content="c1")], scores=[0.1])
    fake_index = AsyncMock(query_chunks=AsyncMock(return_value=expected))
    milvus_vec_adapter._get_and_cache_vector_db_index = AsyncMock(return_value=fake_index)

    response = await milvus_vec_adapter.query_chunks("db1", "my_query", {"param": 1})

    fake_index.query_chunks.assert_awaited_once_with("my_query", {"param": 1})
    assert response is expected


@pytest.mark.asyncio
async def test_query_chunks_missing_db_raises(milvus_vec_adapter):
    milvus_vec_adapter._get_and_cache_vector_db_index = AsyncMock(return_value=None)

    with pytest.raises(ValueError):
        await milvus_vec_adapter.query_chunks("db_missing", "q", None)


@pytest.mark.asyncio
async def test_save_openai_vector_store(milvus_vec_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_db_id": "test_db",
        "embedding_model": "test_model",
    }

    await milvus_vec_adapter._save_openai_vector_store(store_id, openai_vector_store)

    assert openai_vector_store["id"] in milvus_vec_adapter.openai_vector_stores
    assert milvus_vec_adapter.openai_vector_stores[openai_vector_store["id"]] == openai_vector_store


@pytest.mark.asyncio
async def test_update_openai_vector_store(milvus_vec_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_db_id": "test_db",
        "embedding_model": "test_model",
    }

    await milvus_vec_adapter._save_openai_vector_store(store_id, openai_vector_store)
    openai_vector_store["description"] = "Updated description"
    await milvus_vec_adapter._update_openai_vector_store(store_id, openai_vector_store)
    assert milvus_vec_adapter.openai_vector_stores[openai_vector_store["id"]] == openai_vector_store


@pytest.mark.asyncio
async def test_delete_openai_vector_store(milvus_vec_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_db_id": "test_db",
        "embedding_model": "test_model",
    }

    await milvus_vec_adapter._save_openai_vector_store(store_id, openai_vector_store)
    await milvus_vec_adapter._delete_openai_vector_store_from_storage(store_id)
    assert openai_vector_store["id"] not in milvus_vec_adapter.openai_vector_stores


@pytest.mark.asyncio
async def test_load_openai_vector_stores(milvus_vec_adapter):
    store_id = "vs_1234"
    openai_vector_store = {
        "id": store_id,
        "name": "Test Store",
        "description": "A test OpenAI vector store",
        "vector_db_id": "test_db",
        "embedding_model": "test_model",
    }

    await milvus_vec_adapter._save_openai_vector_store(store_id, openai_vector_store)
    loaded_stores = await milvus_vec_adapter._load_openai_vector_stores()
    assert loaded_stores[store_id] == openai_vector_store


@pytest.mark.asyncio
async def test_save_openai_vector_store_file(milvus_vec_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    # validating we don't raise an exception
    await milvus_vec_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)


@pytest.mark.asyncio
async def test_update_openai_vector_store_file(milvus_vec_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await milvus_vec_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    updated_file_info = file_info.copy()
    updated_file_info["filename"] = "updated_test_file.txt"

    await milvus_vec_adapter._update_openai_vector_store_file(
        store_id,
        file_id,
        updated_file_info,
    )

    loaded_contents = await milvus_vec_adapter._load_openai_vector_store_file(store_id, file_id)
    assert loaded_contents == updated_file_info
    assert loaded_contents != file_info


@pytest.mark.asyncio
async def test_load_openai_vector_store_file_contents(milvus_vec_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await milvus_vec_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)

    loaded_contents = await milvus_vec_adapter._load_openai_vector_store_file_contents(store_id, file_id)
    assert loaded_contents == file_contents


@pytest.mark.asyncio
async def test_delete_openai_vector_store_file_from_storage(milvus_vec_adapter, tmp_path_factory):
    store_id = "vs_1234"
    file_id = "file_1234"

    file_info = {
        "id": file_id,
        "status": "completed",
        "vector_store_id": store_id,
        "attributes": {},
        "filename": "test_file.txt",
        "created_at": int(time.time()),
    }

    file_contents = [
        {"content": "Test content", "chunk_metadata": {"chunk_id": "chunk_001"}, "metadata": {"file_id": file_id}}
    ]

    await milvus_vec_adapter._save_openai_vector_store_file(store_id, file_id, file_info, file_contents)
    await milvus_vec_adapter._delete_openai_vector_store_file_from_storage(store_id, file_id)

    loaded_contents = await milvus_vec_adapter._load_openai_vector_store_file_contents(store_id, file_id)
    assert loaded_contents == []
