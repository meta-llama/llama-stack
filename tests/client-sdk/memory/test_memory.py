# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client.types.memory_insert_params import Document


def test_memory_bank(llama_stack_client):
    providers = llama_stack_client.providers.list()
    if "memory" not in providers:
        pytest.skip("No memory provider available")

    # get memory provider id
    assert len(providers["memory"]) > 0

    memory_provider_id = providers["memory"][0].provider_id
    memory_bank_id = "test_bank"

    llama_stack_client.memory_banks.register(
        memory_bank_id=memory_bank_id,
        params={
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

    # add documents to memory bank
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "datasets.rst",
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

    # query documents
    response = llama_stack_client.memory.query(
        bank_id=memory_bank_id,
        query=[
            "How do I use lora",
        ],
    )

    assert len(response.chunks) > 0
    assert len(response.chunks) == len(response.scores)

    contents = [chunk.content for chunk in response.chunks]
    assert "lora" in contents[0].lower()
