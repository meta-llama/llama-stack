# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403

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


async def register_memory_bank(banks_impl: MemoryBanks):
    bank = VectorMemoryBankDef(
        identifier="test_bank",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=512,
        overlap_size_in_tokens=64,
    )

    await banks_impl.register_memory_bank(bank)


class TestMemory:
    @pytest.mark.asyncio
    async def test_banks_list(self, memory_stack):
        # NOTE: this needs you to ensure that you are starting from a clean state
        # but so far we don't have an unregister API unfortunately, so be careful
        _, banks_impl = memory_stack
        response = await banks_impl.list_memory_banks()
        assert isinstance(response, list)
        assert len(response) == 0

    @pytest.mark.asyncio
    async def test_banks_register(self, memory_stack):
        # NOTE: this needs you to ensure that you are starting from a clean state
        # but so far we don't have an unregister API unfortunately, so be careful
        _, banks_impl = memory_stack
        bank = VectorMemoryBankDef(
            identifier="test_bank_no_provider",
            embedding_model="all-MiniLM-L6-v2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        )

        await banks_impl.register_memory_bank(bank)
        response = await banks_impl.list_memory_banks()
        assert isinstance(response, list)
        assert len(response) == 1

        # register same memory bank with same id again will fail
        await banks_impl.register_memory_bank(bank)
        response = await banks_impl.list_memory_banks()
        assert isinstance(response, list)
        assert len(response) == 1
