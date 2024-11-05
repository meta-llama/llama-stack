# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile

import pytest
from llama_stack.apis.memory import MemoryBankType, VectorMemoryBankDef
from llama_stack.providers.impls.meta_reference.memory.config import FaissImplConfig

from llama_stack.providers.impls.meta_reference.memory.faiss import FaissMemoryImpl
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


class TestFaissMemoryImpl:
    @pytest.fixture
    def faiss_impl(self):
        # Create a temporary SQLite database file
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        config = FaissImplConfig(kvstore=SqliteKVStoreConfig(db_path=temp_db.name))
        return FaissMemoryImpl(config)

    @pytest.mark.asyncio
    async def test_initialize(self, faiss_impl):
        # Test empty initialization
        await faiss_impl.initialize()
        assert len(faiss_impl.cache) == 0

        # Test initialization with existing banks
        bank = VectorMemoryBankDef(
            identifier="test_bank",
            type=MemoryBankType.vector.value,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )

        # Register a bank and reinitialize to test loading
        await faiss_impl.register_memory_bank(bank)

        # Create new instance to test initialization with existing data
        new_impl = FaissMemoryImpl(faiss_impl.config)
        await new_impl.initialize()

        assert len(new_impl.cache) == 1
        assert "test_bank" in new_impl.cache

    @pytest.mark.asyncio
    async def test_register_memory_bank(self, faiss_impl):
        bank = VectorMemoryBankDef(
            identifier="test_bank",
            type=MemoryBankType.vector.value,
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )

        await faiss_impl.initialize()
        await faiss_impl.register_memory_bank(bank)

        assert "test_bank" in faiss_impl.cache
        assert faiss_impl.cache["test_bank"].bank == bank

        # Verify persistence
        new_impl = FaissMemoryImpl(faiss_impl.config)
        await new_impl.initialize()
        assert "test_bank" in new_impl.cache


if __name__ == "__main__":
    pytest.main([__file__])
