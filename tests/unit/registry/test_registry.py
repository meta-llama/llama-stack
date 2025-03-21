# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
import pytest_asyncio

from llama_stack.apis.inference import Model
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.distribution.store.registry import (
    KEY_FORMAT,
    CachedDiskDistributionRegistry,
    DiskDistributionRegistry,
)
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


@pytest.fixture
def config():
    config = SqliteKVStoreConfig(db_path="/tmp/test_registry.db")
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
    return config


@pytest_asyncio.fixture(scope="function")
async def registry(config):
    registry = DiskDistributionRegistry(await kvstore_impl(config))
    await registry.initialize()
    return registry


@pytest_asyncio.fixture(scope="function")
async def cached_registry(config):
    registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await registry.initialize()
    return registry


@pytest.fixture
def sample_vector_db():
    return VectorDB(
        identifier="test_vector_db",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_resource_id="test_vector_db",
        provider_id="test-provider",
    )


@pytest.fixture
def sample_model():
    return Model(
        identifier="test_model",
        provider_resource_id="test_model",
        provider_id="test-provider",
    )


@pytest.mark.asyncio
async def test_registry_initialization(registry):
    # Test empty registry
    result = await registry.get("nonexistent", "nonexistent")
    assert result is None


@pytest.mark.asyncio
async def test_basic_registration(registry, sample_vector_db, sample_model):
    print(f"Registering {sample_vector_db}")
    await registry.register(sample_vector_db)
    print(f"Registering {sample_model}")
    await registry.register(sample_model)
    print("Getting vector_db")
    result_vector_db = await registry.get("vector_db", "test_vector_db")
    assert result_vector_db is not None
    assert result_vector_db.identifier == sample_vector_db.identifier
    assert result_vector_db.embedding_model == sample_vector_db.embedding_model
    assert result_vector_db.provider_id == sample_vector_db.provider_id

    result_model = await registry.get("model", "test_model")
    assert result_model is not None
    assert result_model.identifier == sample_model.identifier
    assert result_model.provider_id == sample_model.provider_id


@pytest.mark.asyncio
async def test_cached_registry_initialization(config, sample_vector_db, sample_model):
    # First populate the disk registry
    disk_registry = DiskDistributionRegistry(await kvstore_impl(config))
    await disk_registry.initialize()
    await disk_registry.register(sample_vector_db)
    await disk_registry.register(sample_model)

    # Test cached version loads from disk
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    result_vector_db = await cached_registry.get("vector_db", "test_vector_db")
    assert result_vector_db is not None
    assert result_vector_db.identifier == sample_vector_db.identifier
    assert result_vector_db.embedding_model == sample_vector_db.embedding_model
    assert result_vector_db.embedding_dimension == sample_vector_db.embedding_dimension
    assert result_vector_db.provider_id == sample_vector_db.provider_id


@pytest.mark.asyncio
async def test_cached_registry_updates(config):
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    new_vector_db = VectorDB(
        identifier="test_vector_db_2",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_resource_id="test_vector_db_2",
        provider_id="baz",
    )
    await cached_registry.register(new_vector_db)

    # Verify in cache
    result_vector_db = await cached_registry.get("vector_db", "test_vector_db_2")
    assert result_vector_db is not None
    assert result_vector_db.identifier == new_vector_db.identifier
    assert result_vector_db.provider_id == new_vector_db.provider_id

    # Verify persisted to disk
    new_registry = DiskDistributionRegistry(await kvstore_impl(config))
    await new_registry.initialize()
    result_vector_db = await new_registry.get("vector_db", "test_vector_db_2")
    assert result_vector_db is not None
    assert result_vector_db.identifier == new_vector_db.identifier
    assert result_vector_db.provider_id == new_vector_db.provider_id


@pytest.mark.asyncio
async def test_duplicate_provider_registration(config):
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    original_vector_db = VectorDB(
        identifier="test_vector_db_2",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_resource_id="test_vector_db_2",
        provider_id="baz",
    )
    await cached_registry.register(original_vector_db)

    duplicate_vector_db = VectorDB(
        identifier="test_vector_db_2",
        embedding_model="different-model",
        embedding_dimension=384,
        provider_resource_id="test_vector_db_2",
        provider_id="baz",  # Same provider_id
    )
    await cached_registry.register(duplicate_vector_db)

    result = await cached_registry.get("vector_db", "test_vector_db_2")
    assert result is not None
    assert result.embedding_model == original_vector_db.embedding_model  # Original values preserved


@pytest.mark.asyncio
async def test_get_all_objects(config):
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    # Create multiple test banks
    test_vector_dbs = [
        VectorDB(
            identifier=f"test_vector_db_{i}",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_resource_id=f"test_vector_db_{i}",
            provider_id=f"provider_{i}",
        )
        for i in range(3)
    ]

    # Register all vector_dbs
    for vector_db in test_vector_dbs:
        await cached_registry.register(vector_db)

    # Test get_all retrieval
    all_results = await cached_registry.get_all()
    assert len(all_results) == 3

    # Verify each vector_db was stored correctly
    for original_vector_db in test_vector_dbs:
        matching_vector_dbs = [v for v in all_results if v.identifier == original_vector_db.identifier]
        assert len(matching_vector_dbs) == 1
        stored_vector_db = matching_vector_dbs[0]
        assert stored_vector_db.embedding_model == original_vector_db.embedding_model
        assert stored_vector_db.provider_id == original_vector_db.provider_id
        assert stored_vector_db.embedding_dimension == original_vector_db.embedding_dimension


@pytest.mark.asyncio
async def test_parse_registry_values_error_handling(config):
    kvstore = await kvstore_impl(config)

    valid_db = VectorDB(
        identifier="valid_vector_db",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_resource_id="valid_vector_db",
        provider_id="test-provider",
    )

    await kvstore.set(KEY_FORMAT.format(type="vector_db", identifier="valid_vector_db"), valid_db.model_dump_json())

    await kvstore.set(KEY_FORMAT.format(type="vector_db", identifier="corrupted_json"), "{not valid json")

    await kvstore.set(
        KEY_FORMAT.format(type="vector_db", identifier="missing_fields"),
        '{"type": "vector_db", "identifier": "missing_fields"}',
    )

    test_registry = DiskDistributionRegistry(kvstore)
    await test_registry.initialize()

    # Get all objects, which should only return the valid one
    all_objects = await test_registry.get_all()

    # Should have filtered out the invalid entries
    assert len(all_objects) == 1
    assert all_objects[0].identifier == "valid_vector_db"

    # Check that the get method also handles errors correctly
    invalid_obj = await test_registry.get("vector_db", "corrupted_json")
    assert invalid_obj is None

    invalid_obj = await test_registry.get("vector_db", "missing_fields")
    assert invalid_obj is None


@pytest.mark.asyncio
async def test_cached_registry_error_handling(config):
    kvstore = await kvstore_impl(config)

    valid_db = VectorDB(
        identifier="valid_cached_db",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_resource_id="valid_cached_db",
        provider_id="test-provider",
    )

    await kvstore.set(KEY_FORMAT.format(type="vector_db", identifier="valid_cached_db"), valid_db.model_dump_json())

    await kvstore.set(
        KEY_FORMAT.format(type="vector_db", identifier="invalid_cached_db"),
        '{"type": "vector_db", "identifier": "invalid_cached_db", "embedding_model": 12345}',  # Should be string
    )

    cached_registry = CachedDiskDistributionRegistry(kvstore)
    await cached_registry.initialize()

    all_objects = await cached_registry.get_all()
    assert len(all_objects) == 1
    assert all_objects[0].identifier == "valid_cached_db"

    invalid_obj = await cached_registry.get("vector_db", "invalid_cached_db")
    assert invalid_obj is None
