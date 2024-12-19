# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from datetime import datetime, timedelta

import pytest

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.apis.memory_banks.memory_banks import VectorMemoryBankParams
from llama_stack.providers.utils.kvstore import RedisKVStoreConfig, kvstore_impl

# How to run this test:
#
# pytest llama_stack/providers/tests/memory/test_memory.py
#   -m "meta_reference"
#   -v -s --tb=short --disable-warnings


@pytest.fixture
def sample_documents():
    return [
        MemoryBankDocument(
            document_id="doc1",
            content="Python is a high-level programming language.",
            metadata={"category": "programming", "difficulty": "beginner"},
        ),
        MemoryBankDocument(
            document_id="doc2",
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "AI", "difficulty": "advanced"},
        ),
        MemoryBankDocument(
            document_id="doc3",
            content="Data structures are fundamental to computer science.",
            metadata={"category": "computer science", "difficulty": "intermediate"},
        ),
        MemoryBankDocument(
            document_id="doc4",
            content="Neural networks are inspired by biological neural networks.",
            metadata={"category": "AI", "difficulty": "advanced"},
        ),
    ]


@pytest.fixture
def redis_config():
    return RedisKVStoreConfig(host="localhost", port=6379)


@pytest_asyncio.fixture
async def redis_kvstore(redis_config):
    kvstore = await kvstore_impl(redis_config)
    await kvstore.initialize()
    return kvstore


async def register_memory_bank(
    banks_impl: MemoryBanks, embedding_model: str
) -> MemoryBank:
    bank_id = f"test_bank_{uuid.uuid4().hex}"
    return await banks_impl.register_memory_bank(
        memory_bank_id=bank_id,
        params=VectorMemoryBankParams(
            embedding_model=embedding_model,
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        ),
    )


class TestMemory:
    @pytest.mark.asyncio
    async def test_banks_list(self, memory_stack, embedding_model):
        _, banks_impl = memory_stack

        # Register a test bank
        registered_bank = await register_memory_bank(banks_impl, embedding_model)

        try:
            # Verify our bank shows up in list
            response = await banks_impl.list_memory_banks()
            assert isinstance(response, list)
            assert any(
                bank.memory_bank_id == registered_bank.memory_bank_id
                for bank in response
            )
        finally:
            # Clean up
            await banks_impl.unregister_memory_bank(registered_bank.memory_bank_id)

        # Verify our bank was removed
        response = await banks_impl.list_memory_banks()
        assert all(
            bank.memory_bank_id != registered_bank.memory_bank_id for bank in response
        )

    @pytest.mark.asyncio
    async def test_banks_register(self, memory_stack, embedding_model):
        _, banks_impl = memory_stack

        bank_id = f"test_bank_{uuid.uuid4().hex}"

        try:
            # Register initial bank
            await banks_impl.register_memory_bank(
                memory_bank_id=bank_id,
                params=VectorMemoryBankParams(
                    embedding_model=embedding_model,
                    chunk_size_in_tokens=512,
                    overlap_size_in_tokens=64,
                ),
            )

            # Verify our bank exists
            response = await banks_impl.list_memory_banks()
            assert isinstance(response, list)
            assert any(bank.memory_bank_id == bank_id for bank in response)

            # Try registering same bank again
            await banks_impl.register_memory_bank(
                memory_bank_id=bank_id,
                params=VectorMemoryBankParams(
                    embedding_model=embedding_model,
                    chunk_size_in_tokens=512,
                    overlap_size_in_tokens=64,
                ),
            )

            # Verify still only one instance of our bank
            response = await banks_impl.list_memory_banks()
            assert isinstance(response, list)
            assert (
                len([bank for bank in response if bank.memory_bank_id == bank_id]) == 1
            )
        finally:
            # Clean up
            await banks_impl.unregister_memory_bank(bank_id)

    @pytest.mark.asyncio
    async def test_query_documents(
        self, memory_stack, embedding_model, sample_documents
    ):
        memory_impl, banks_impl = memory_stack

        with pytest.raises(ValueError):
            await memory_impl.insert_documents("test_bank", sample_documents)

        registered_bank = await register_memory_bank(banks_impl, embedding_model)
        await memory_impl.insert_documents(
            registered_bank.memory_bank_id, sample_documents
        )

        query1 = "programming language"
        response1 = await memory_impl.query_documents(
            registered_bank.memory_bank_id, query1
        )
        assert_valid_response(response1)
        assert any("Python" in chunk.content for chunk in response1.chunks)

        # Test case 3: Query with semantic similarity
        query3 = "AI and brain-inspired computing"
        response3 = await memory_impl.query_documents(
            registered_bank.memory_bank_id, query3
        )
        assert_valid_response(response3)
        assert any(
            "neural networks" in chunk.content.lower() for chunk in response3.chunks
        )

        # Test case 4: Query with limit on number of results
        query4 = "computer"
        params4 = {"max_chunks": 2}
        response4 = await memory_impl.query_documents(
            registered_bank.memory_bank_id, query4, params4
        )
        assert_valid_response(response4)
        assert len(response4.chunks) <= 2

        # Test case 5: Query with threshold on similarity score
        query5 = "quantum computing"  # Not directly related to any document
        params5 = {"score_threshold": 0.01}
        response5 = await memory_impl.query_documents(
            registered_bank.memory_bank_id, query5, params5
        )
        assert_valid_response(response5)
        print("The scores are:", response5.scores)
        assert all(score >= 0.01 for score in response5.scores)

    @pytest.mark.asyncio
    async def test_redis_kvstore_initialization(self, redis_kvstore):
        # Test empty Redis KVStore
        result = await redis_kvstore.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_kvstore_crud_operations(self, redis_kvstore):
        # Test set and get operations
        await redis_kvstore.set("key1", "value1")
        result = await redis_kvstore.get("key1")
        assert result == "value1"

        # Test delete operation
        await redis_kvstore.delete("key1")
        result = await redis_kvstore.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_kvstore_namespaced_keys(self, redis_config):
        redis_config.namespace = "test_namespace"
        kvstore = await kvstore_impl(redis_config)
        await kvstore.initialize()

        # Test namespaced set and get operations
        await kvstore.set("key1", "value1")
        result = await kvstore.get("key1")
        assert result == "value1"

        # Test delete operation with namespaced key
        await kvstore.delete("key1")
        result = await kvstore.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_kvstore_expiration_handling(self, redis_kvstore):
        # Test set operation with expiration
        expiration = datetime.now() + timedelta(seconds=1)
        await redis_kvstore.set("key1", "value1", expiration=expiration)

        # Verify key is retrievable before expiration
        result = await redis_kvstore.get("key1")
        assert result == "value1"

        # Wait for expiration and verify key is no longer retrievable
        await asyncio.sleep(2)
        result = await redis_kvstore.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_redis_kvstore_error_handling(self, redis_config):
        # Test connection to invalid Redis server
        invalid_config = RedisKVStoreConfig(host="invalid_host", port=6379)
        with pytest.raises(RuntimeError):
            await kvstore_impl(invalid_config)

    @pytest.mark.asyncio
    async def test_redis_kvstore_concurrency(self, redis_kvstore):
        # Test concurrent access to Redis KVStore
        async def set_value(key, value):
            await redis_kvstore.set(key, value)

        async def get_value(key):
            return await redis_kvstore.get(key)

        tasks = []
        for i in range(10):
            tasks.append(set_value(f"key{i}", f"value{i}"))
        await asyncio.gather(*tasks)

        tasks = []
        for i in range(10):
            tasks.append(get_value(f"key{i}"))
        results = await asyncio.gather(*tasks)

        for i in range(10):
            assert results[i] == f"value{i}"


def assert_valid_response(response: QueryDocumentsResponse):
    assert isinstance(response, QueryDocumentsResponse)
    assert len(response.chunks) > 0
    assert len(response.scores) > 0
    assert len(response.chunks) == len(response.scores)
    for chunk in response.chunks:
        assert isinstance(chunk.content, str)
        assert chunk.document_id is not None
