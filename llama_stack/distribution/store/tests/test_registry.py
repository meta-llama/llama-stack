# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
import pytest_asyncio
from llama_stack.apis.inference import Model
from llama_stack.apis.memory_banks import VectorMemoryBank

from llama_stack.distribution.store.registry import (
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
def sample_bank():
    return VectorMemoryBank(
        identifier="test_bank",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=512,
        overlap_size_in_tokens=64,
        provider_resource_id="test_bank",
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
async def test_basic_registration(registry, sample_bank, sample_model):
    print(f"Registering {sample_bank}")
    await registry.register(sample_bank)
    print(f"Registering {sample_model}")
    await registry.register(sample_model)
    print("Getting bank")
    result_bank = await registry.get("memory_bank", "test_bank")
    assert result_bank is not None
    assert result_bank.identifier == sample_bank.identifier
    assert result_bank.embedding_model == sample_bank.embedding_model
    assert result_bank.chunk_size_in_tokens == sample_bank.chunk_size_in_tokens
    assert result_bank.overlap_size_in_tokens == sample_bank.overlap_size_in_tokens
    assert result_bank.provider_id == sample_bank.provider_id

    result_model = await registry.get("model", "test_model")
    assert result_model is not None
    assert result_model.identifier == sample_model.identifier
    assert result_model.provider_id == sample_model.provider_id


@pytest.mark.asyncio
async def test_cached_registry_initialization(config, sample_bank, sample_model):
    # First populate the disk registry
    disk_registry = DiskDistributionRegistry(await kvstore_impl(config))
    await disk_registry.initialize()
    await disk_registry.register(sample_bank)
    await disk_registry.register(sample_model)

    # Test cached version loads from disk
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    result_bank = await cached_registry.get("memory_bank", "test_bank")
    assert result_bank is not None
    assert result_bank.identifier == sample_bank.identifier
    assert result_bank.embedding_model == sample_bank.embedding_model
    assert result_bank.chunk_size_in_tokens == sample_bank.chunk_size_in_tokens
    assert result_bank.overlap_size_in_tokens == sample_bank.overlap_size_in_tokens
    assert result_bank.provider_id == sample_bank.provider_id


@pytest.mark.asyncio
async def test_cached_registry_updates(config):
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    new_bank = VectorMemoryBank(
        identifier="test_bank_2",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=256,
        overlap_size_in_tokens=32,
        provider_resource_id="test_bank_2",
        provider_id="baz",
    )
    await cached_registry.register(new_bank)

    # Verify in cache
    result_bank = await cached_registry.get("memory_bank", "test_bank_2")
    assert result_bank is not None
    assert result_bank.identifier == new_bank.identifier
    assert result_bank.provider_id == new_bank.provider_id

    # Verify persisted to disk
    new_registry = DiskDistributionRegistry(await kvstore_impl(config))
    await new_registry.initialize()
    result_bank = await new_registry.get("memory_bank", "test_bank_2")
    assert result_bank is not None
    assert result_bank.identifier == new_bank.identifier
    assert result_bank.provider_id == new_bank.provider_id


@pytest.mark.asyncio
async def test_duplicate_provider_registration(config):
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    original_bank = VectorMemoryBank(
        identifier="test_bank_2",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=256,
        overlap_size_in_tokens=32,
        provider_resource_id="test_bank_2",
        provider_id="baz",
    )
    await cached_registry.register(original_bank)

    duplicate_bank = VectorMemoryBank(
        identifier="test_bank_2",
        embedding_model="different-model",
        chunk_size_in_tokens=128,
        overlap_size_in_tokens=16,
        provider_resource_id="test_bank_2",
        provider_id="baz",  # Same provider_id
    )
    await cached_registry.register(duplicate_bank)

    result = await cached_registry.get("memory_bank", "test_bank_2")
    assert result is not None
    assert (
        result.embedding_model == original_bank.embedding_model
    )  # Original values preserved


@pytest.mark.asyncio
async def test_get_all_objects(config):
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    # Create multiple test banks
    test_banks = [
        VectorMemoryBank(
            identifier=f"test_bank_{i}",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=256,
            overlap_size_in_tokens=32,
            provider_resource_id=f"test_bank_{i}",
            provider_id=f"provider_{i}",
        )
        for i in range(3)
    ]

    # Register all banks
    for bank in test_banks:
        await cached_registry.register(bank)

    # Test get_all retrieval
    all_results = await cached_registry.get_all()
    assert len(all_results) == 3

    # Verify each bank was stored correctly
    for original_bank in test_banks:
        matching_banks = [
            b for b in all_results if b.identifier == original_bank.identifier
        ]
        assert len(matching_banks) == 1
        stored_bank = matching_banks[0]
        assert stored_bank.embedding_model == original_bank.embedding_model
        assert stored_bank.provider_id == original_bank.provider_id
        assert stored_bank.chunk_size_in_tokens == original_bank.chunk_size_in_tokens
        assert (
            stored_bank.overlap_size_in_tokens == original_bank.overlap_size_in_tokens
        )
