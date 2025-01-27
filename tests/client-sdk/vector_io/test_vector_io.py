# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import random

import pytest


@pytest.fixture(scope="function")
def empty_vector_db_registry(llama_stack_client):
    vector_dbs = [
        vector_db.identifier for vector_db in llama_stack_client.vector_dbs.list()
    ]
    for vector_db_id in vector_dbs:
        llama_stack_client.vector_dbs.unregister(vector_db_id=vector_db_id)


@pytest.fixture(scope="function")
def single_entry_vector_db_registry(llama_stack_client, empty_vector_db_registry):
    vector_db_id = f"test_vector_db_{random.randint(1000, 9999)}"
    llama_stack_client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_id="faiss",
    )
    vector_dbs = [
        vector_db.identifier for vector_db in llama_stack_client.vector_dbs.list()
    ]
    return vector_dbs


def test_vector_db_retrieve(
    llama_stack_client, embedding_model, empty_vector_db_registry
):
    # Register a memory bank first
    vector_db_id = f"test_vector_db_{random.randint(1000, 9999)}"
    llama_stack_client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model,
        embedding_dimension=384,
        provider_id="faiss",
    )

    # Retrieve the memory bank and validate its properties
    response = llama_stack_client.vector_dbs.retrieve(vector_db_id=vector_db_id)
    assert response is not None
    assert response.identifier == vector_db_id
    assert response.embedding_model == embedding_model
    assert response.provider_id == "faiss"
    assert response.provider_resource_id == vector_db_id


def test_vector_db_list(llama_stack_client, empty_vector_db_registry):
    vector_dbs_after_register = [
        vector_db.identifier for vector_db in llama_stack_client.vector_dbs.list()
    ]
    assert len(vector_dbs_after_register) == 0


def test_vector_db_register(
    llama_stack_client, embedding_model, empty_vector_db_registry
):
    vector_db_id = f"test_vector_db_{random.randint(1000, 9999)}"
    llama_stack_client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model,
        embedding_dimension=384,
        provider_id="faiss",
    )

    vector_dbs_after_register = [
        vector_db.identifier for vector_db in llama_stack_client.vector_dbs.list()
    ]
    assert vector_dbs_after_register == [vector_db_id]


def test_vector_db_unregister(llama_stack_client, single_entry_vector_db_registry):
    vector_dbs = [
        vector_db.identifier for vector_db in llama_stack_client.vector_dbs.list()
    ]
    assert len(vector_dbs) == 1

    vector_db_id = vector_dbs[0]
    llama_stack_client.vector_dbs.unregister(vector_db_id=vector_db_id)

    vector_dbs = [
        vector_db.identifier for vector_db in llama_stack_client.vector_dbs.list()
    ]
    assert len(vector_dbs) == 0
