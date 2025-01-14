# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import random

import pytest
from llama_stack.apis.memory import MemoryBankDocument

from llama_stack_client.types.memory_insert_params import Document


@pytest.fixture(scope="function")
def empty_memory_bank_registry(llama_stack_client):
    memory_banks = [
        memory_bank.identifier for memory_bank in llama_stack_client.memory_banks.list()
    ]
    for memory_bank_id in memory_banks:
        llama_stack_client.memory_banks.unregister(memory_bank_id=memory_bank_id)


@pytest.fixture(scope="function")
def single_entry_memory_bank_registry(llama_stack_client, empty_memory_bank_registry):
    memory_bank_id = f"test_bank_{random.randint(1000, 9999)}"
    llama_stack_client.memory_banks.register(
        memory_bank_id=memory_bank_id,
        params={
            "memory_bank_type": "vector",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
        },
        provider_id="faiss",
    )
    memory_banks = [
        memory_bank.identifier for memory_bank in llama_stack_client.memory_banks.list()
    ]
    return memory_banks


@pytest.fixture(scope="session")
def sample_documents():
    return [
        MemoryBankDocument(
            document_id="test-doc-1",
            content="Python is a high-level programming language.",
            metadata={"category": "programming", "difficulty": "beginner"},
        ),
        MemoryBankDocument(
            document_id="test-doc-2",
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "AI", "difficulty": "advanced"},
        ),
        MemoryBankDocument(
            document_id="test-doc-3",
            content="Data structures are fundamental to computer science.",
            metadata={"category": "computer science", "difficulty": "intermediate"},
        ),
        MemoryBankDocument(
            document_id="test-doc-4",
            content="Neural networks are inspired by biological neural networks.",
            metadata={"category": "AI", "difficulty": "advanced"},
        ),
    ]


def assert_valid_response(response):
    assert len(response.chunks) > 0
    assert len(response.scores) > 0
    assert len(response.chunks) == len(response.scores)
    for chunk in response.chunks:
        assert isinstance(chunk.content, str)
        assert chunk.document_id is not None


def test_memory_bank_retrieve(llama_stack_client, empty_memory_bank_registry):
    # Register a memory bank first
    memory_bank_id = f"test_bank_{random.randint(1000, 9999)}"
    llama_stack_client.memory_banks.register(
        memory_bank_id=memory_bank_id,
        params={
            "memory_bank_type": "vector",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
        },
        provider_id="faiss",
    )

    # Retrieve the memory bank and validate its properties
    response = llama_stack_client.memory_banks.retrieve(memory_bank_id=memory_bank_id)
    assert response is not None
    assert response.identifier == memory_bank_id
    assert response.type == "memory_bank"
    assert response.memory_bank_type == "vector"
    assert response.embedding_model == "all-MiniLM-L6-v2"
    assert response.chunk_size_in_tokens == 512
    assert response.overlap_size_in_tokens == 64
    assert response.provider_id == "faiss"
    assert response.provider_resource_id == memory_bank_id


def test_memory_bank_list(llama_stack_client, empty_memory_bank_registry):
    memory_banks_after_register = [
        memory_bank.identifier for memory_bank in llama_stack_client.memory_banks.list()
    ]
    assert len(memory_banks_after_register) == 0


def test_memory_bank_register(llama_stack_client, empty_memory_bank_registry):
    memory_provider_id = "faiss"
    memory_bank_id = f"test_bank_{random.randint(1000, 9999)}"
    llama_stack_client.memory_banks.register(
        memory_bank_id=memory_bank_id,
        params={
            "memory_bank_type": "vector",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
        },
        provider_id=memory_provider_id,
    )

    memory_banks_after_register = [
        memory_bank.identifier for memory_bank in llama_stack_client.memory_banks.list()
    ]
    assert memory_banks_after_register == [memory_bank_id]


def test_memory_bank_unregister(llama_stack_client, single_entry_memory_bank_registry):
    memory_banks = [
        memory_bank.identifier for memory_bank in llama_stack_client.memory_banks.list()
    ]
    assert len(memory_banks) == 1

    memory_bank_id = memory_banks[0]
    llama_stack_client.memory_banks.unregister(memory_bank_id=memory_bank_id)

    memory_banks = [
        memory_bank.identifier for memory_bank in llama_stack_client.memory_banks.list()
    ]
    assert len(memory_banks) == 0


def test_memory_bank_insert_inline_and_query(
    llama_stack_client, single_entry_memory_bank_registry, sample_documents
):
    memory_bank_id = single_entry_memory_bank_registry[0]
    llama_stack_client.memory.insert(
        bank_id=memory_bank_id,
        documents=sample_documents,
    )

    # Query with a direct match
    query1 = "programming language"
    response1 = llama_stack_client.memory.query(
        bank_id=memory_bank_id,
        query=query1,
    )
    assert_valid_response(response1)
    assert any("Python" in chunk.content for chunk in response1.chunks)

    # Query with semantic similarity
    query2 = "AI and brain-inspired computing"
    response2 = llama_stack_client.memory.query(
        bank_id=memory_bank_id,
        query=query2,
    )
    assert_valid_response(response2)
    assert any("neural networks" in chunk.content.lower() for chunk in response2.chunks)

    # Query with limit on number of results (max_chunks=2)
    query3 = "computer"
    response3 = llama_stack_client.memory.query(
        bank_id=memory_bank_id,
        query=query3,
        params={"max_chunks": 2},
    )
    assert_valid_response(response3)
    assert len(response3.chunks) <= 2

    # Query with threshold on similarity score
    query4 = "computer"
    response4 = llama_stack_client.memory.query(
        bank_id=memory_bank_id,
        query=query4,
        params={"score_threshold": 0.01},
    )
    assert_valid_response(response4)
    assert all(score >= 0.01 for score in response4.scores)


def test_memory_bank_insert_from_url_and_query(
    llama_stack_client, empty_memory_bank_registry
):
    providers = llama_stack_client.providers.list()
    assert "memory" in providers
    assert len(providers["memory"]) > 0

    memory_provider_id = providers["memory"][0].provider_id
    memory_bank_id = "test_bank"

    llama_stack_client.memory_banks.register(
        memory_bank_id=memory_bank_id,
        params={
            "memory_bank_type": "vector",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
        },
        provider_id=memory_provider_id,
    )

    # list to check memory bank is successfully registered
    available_memory_banks = [
        memory_bank.identifier for memory_bank in llama_stack_client.memory_banks.list()
    ]
    assert memory_bank_id in available_memory_banks

    # URLs of documents to insert
    # TODO: Move to test/memory/resources then update the url to
    # https://raw.githubusercontent.com/meta-llama/llama-stack/main/tests/memory/resources/{url}
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
    ]
    documents = [
        Document(
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={},
        )
        for i, url in enumerate(urls)
    ]

    llama_stack_client.memory.insert(
        bank_id=memory_bank_id,
        documents=documents,
    )

    # Query for the name of method
    response1 = llama_stack_client.memory.query(
        bank_id=memory_bank_id,
        query="What's the name of the fine-tunning method used?",
    )
    assert_valid_response(response1)
    assert any("lora" in chunk.content.lower() for chunk in response1.chunks)

    # Query for the name of model
    response2 = llama_stack_client.memory.query(
        bank_id=memory_bank_id,
        query="Which Llama model is mentioned?",
    )
    assert_valid_response(response1)
    assert any("llama2" in chunk.content.lower() for chunk in response2.chunks)
