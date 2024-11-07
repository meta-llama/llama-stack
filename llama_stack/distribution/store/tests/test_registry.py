# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest
import pytest_asyncio
from llama_stack.distribution.store import *  # noqa F403
from llama_stack.apis.inference import ModelDefWithProvider
from llama_stack.apis.memory_banks import VectorMemoryBankDef
from llama_stack.providers.utils.kvstore import kvstore_impl, SqliteKVStoreConfig
from llama_stack.distribution.datatypes import *  # noqa F403


@pytest.fixture
def config():
    config = SqliteKVStoreConfig(db_path="/tmp/test_registry.db")
    if os.path.exists(config.db_path):
        os.remove(config.db_path)
    return config


@pytest_asyncio.fixture
async def registry(config):
    registry = DiskDistributionRegistry(await kvstore_impl(config))
    await registry.initialize()
    return registry


@pytest_asyncio.fixture
async def cached_registry(config):
    registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await registry.initialize()
    return registry


@pytest.fixture
def sample_bank():
    return VectorMemoryBankDef(
        identifier="test_bank",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=512,
        overlap_size_in_tokens=64,
        provider_id="test-provider",
    )


@pytest.fixture
def sample_model():
    return ModelDefWithProvider(
        identifier="test_model",
        llama_model="Llama3.2-3B-Instruct",
        provider_id="test-provider",
    )


@pytest.mark.asyncio
async def test_registry_initialization(registry):
    # Test empty registry
    results = await registry.get("nonexistent")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_basic_registration(registry, sample_bank, sample_model):
    print(f"Registering {sample_bank}")
    await registry.register(sample_bank)
    print(f"Registering {sample_model}")
    await registry.register(sample_model)
    print("Getting bank")
    results = await registry.get("test_bank")
    assert len(results) == 1
    result_bank = results[0]
    assert result_bank.identifier == sample_bank.identifier
    assert result_bank.embedding_model == sample_bank.embedding_model
    assert result_bank.chunk_size_in_tokens == sample_bank.chunk_size_in_tokens
    assert result_bank.overlap_size_in_tokens == sample_bank.overlap_size_in_tokens
    assert result_bank.provider_id == sample_bank.provider_id

    results = await registry.get("test_model")
    assert len(results) == 1
    result_model = results[0]
    assert result_model.identifier == sample_model.identifier
    assert result_model.llama_model == sample_model.llama_model
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

    results = await cached_registry.get("test_bank")
    assert len(results) == 1
    result_bank = results[0]
    assert result_bank.identifier == sample_bank.identifier
    assert result_bank.embedding_model == sample_bank.embedding_model
    assert result_bank.chunk_size_in_tokens == sample_bank.chunk_size_in_tokens
    assert result_bank.overlap_size_in_tokens == sample_bank.overlap_size_in_tokens
    assert result_bank.provider_id == sample_bank.provider_id


@pytest.mark.asyncio
async def test_cached_registry_updates(config):
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    new_bank = VectorMemoryBankDef(
        identifier="test_bank_2",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=256,
        overlap_size_in_tokens=32,
        provider_id="baz",
    )
    await cached_registry.register(new_bank)

    # Verify in cache
    results = await cached_registry.get("test_bank_2")
    assert len(results) == 1
    result_bank = results[0]
    assert result_bank.identifier == new_bank.identifier
    assert result_bank.provider_id == new_bank.provider_id

    # Verify persisted to disk
    new_registry = DiskDistributionRegistry(await kvstore_impl(config))
    await new_registry.initialize()
    results = await new_registry.get("test_bank_2")
    assert len(results) == 1
    result_bank = results[0]
    assert result_bank.identifier == new_bank.identifier
    assert result_bank.provider_id == new_bank.provider_id


@pytest.mark.asyncio
async def test_duplicate_provider_registration(config):
    cached_registry = CachedDiskDistributionRegistry(await kvstore_impl(config))
    await cached_registry.initialize()

    original_bank = VectorMemoryBankDef(
        identifier="test_bank_2",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=256,
        overlap_size_in_tokens=32,
        provider_id="baz",
    )
    await cached_registry.register(original_bank)

    duplicate_bank = VectorMemoryBankDef(
        identifier="test_bank_2",
        embedding_model="different-model",
        chunk_size_in_tokens=128,
        overlap_size_in_tokens=16,
        provider_id="baz",  # Same provider_id
    )
    await cached_registry.register(duplicate_bank)

    results = await cached_registry.get("test_bank_2")
    assert len(results) == 1  # Still only one result
    assert (
        results[0].embedding_model == original_bank.embedding_model
    )  # Original values preserved
