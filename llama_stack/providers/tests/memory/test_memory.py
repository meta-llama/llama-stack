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

    @pytest.mark.asyncio
    async def test_query_documents(self, memory_stack, sample_documents):
        memory_impl, banks_impl = memory_stack

        with pytest.raises(ValueError):
            await memory_impl.insert_documents("test_bank", sample_documents)

        await register_memory_bank(banks_impl)
        await memory_impl.insert_documents("test_bank", sample_documents)

        query1 = "programming language"
        response1 = await memory_impl.query_documents("test_bank", query1)
        assert_valid_response(response1)
        assert any("Python" in chunk.content for chunk in response1.chunks)

        # Test case 3: Query with semantic similarity
        query3 = "AI and brain-inspired computing"
        response3 = await memory_impl.query_documents("test_bank", query3)
        assert_valid_response(response3)
        assert any(
            "neural networks" in chunk.content.lower() for chunk in response3.chunks
        )

        # Test case 4: Query with limit on number of results
        query4 = "computer"
        params4 = {"max_chunks": 2}
        response4 = await memory_impl.query_documents("test_bank", query4, params4)
        assert_valid_response(response4)
        assert len(response4.chunks) <= 2

        # Test case 5: Query with threshold on similarity score
        query5 = "quantum computing"  # Not directly related to any document
        params5 = {"score_threshold": 0.2}
        response5 = await memory_impl.query_documents("test_bank", query5, params5)
        assert_valid_response(response5)
        print("The scores are:", response5.scores)
        assert all(score >= 0.2 for score in response5.scores)


def assert_valid_response(response: QueryDocumentsResponse):
    assert isinstance(response, QueryDocumentsResponse)
    assert len(response.chunks) > 0
    assert len(response.scores) > 0
    assert len(response.chunks) == len(response.scores)
    for chunk in response.chunks:
        assert isinstance(chunk.content, str)
        assert chunk.document_id is not None
