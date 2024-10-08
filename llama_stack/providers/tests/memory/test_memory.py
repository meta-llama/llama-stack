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


@pytest_asyncio.fixture(scope="session")
async def memory_impl():
    impls = await resolve_impls_for_test(
        Api.memory,
        memory_banks=[],
    )
    return impls[Api.memory]


@pytest.fixture
def sample_document():
    return MemoryBankDocument(
        document_id="doc1",
        content="This is a sample document for testing.",
        mime_type="text/plain",
        metadata={"author": "Test Author"},
    )


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
async def test_query_documents(memory_impl, sample_document):
    with pytest.raises(ValueError):
        await memory_impl.insert_documents("test_bank", [sample_document])

    await register_memory_bank(memory_impl)
    await memory_impl.insert_documents("test_bank", [sample_document])

    query = ["sample ", "document"]
    response = await memory_impl.query_documents("test_bank", query)

    assert isinstance(response, QueryDocumentsResponse)
    assert len(response.chunks) > 0
    assert len(response.scores) > 0
    assert len(response.chunks) == len(response.scores)
