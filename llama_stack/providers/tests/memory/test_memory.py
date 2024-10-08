# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.providers.tests.resolver import resolve_impls_for_test

# How to run this test:
#
# 1. Ensure you have a conda with the right dependencies installed. This is a bit tricky
#    since it depends on the provider you are testing. On top of that you need
#    `pytest` and `pytest-asyncio` installed.
#
# 2. Copy and modify the provider_config_example.yaml depending on the provider you are testing.
#
# 3. Run:
#
# ```bash
# PROVIDER_ID=<your_provider> \
#   PROVIDER_CONFIG=provider_config.yaml \
#   pytest -s llama_stack/providers/tests/memory/test_memory.py \
#   --tb=short --disable-warnings
# ```


@pytest_asyncio.fixture(scope="session")
async def memory_impl():
    impls = await resolve_impls_for_test(
        Api.memory,
        memory_banks=[],
    )
    return impls[Api.memory]


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


async def register_memory_bank(memory_impl: Memory):
    bank = VectorMemoryBankDef(
        identifier="test_bank",
        provider_id="",
        embedding_model="all-MiniLM-L6-v2",
        chunk_size_in_tokens=512,
        overlap_size_in_tokens=64,
    )

    await memory_impl.register_memory_bank(bank)


@pytest.mark.asyncio
async def test_query_documents(memory_impl, sample_documents):
    with pytest.raises(ValueError):
        await memory_impl.insert_documents("test_bank", sample_documents)

    await register_memory_bank(memory_impl)
    await memory_impl.insert_documents("test_bank", sample_documents)

    query1 = "programming language"
    response1 = await memory_impl.query_documents("test_bank", query1)
    assert_valid_response(response1)
    assert any("Python" in chunk.content for chunk in response1.chunks)

    # Test case 3: Query with semantic similarity
    query3 = "AI and brain-inspired computing"
    response3 = await memory_impl.query_documents("test_bank", query3)
    assert_valid_response(response3)
    assert any("neural networks" in chunk.content.lower() for chunk in response3.chunks)

    # Test case 4: Query with limit on number of results
    query4 = "computer"
    params4 = {"max_chunks": 2}
    response4 = await memory_impl.query_documents("test_bank", query4, params4)
    assert_valid_response(response4)
    assert len(response4.chunks) <= 2

    # Test case 5: Query with threshold on similarity score
    query5 = "quantum computing"  # Not directly related to any document
    params5 = {"score_threshold": 0.5}
    response5 = await memory_impl.query_documents("test_bank", query5, params5)
    assert_valid_response(response5)
    assert all(score >= 0.5 for score in response5.scores)


def assert_valid_response(response: QueryDocumentsResponse):
    assert isinstance(response, QueryDocumentsResponse)
    assert len(response.chunks) > 0
    assert len(response.scores) > 0
    assert len(response.chunks) == len(response.scores)
    for chunk in response.chunks:
        assert isinstance(chunk.content, str)
        assert chunk.document_id is not None
