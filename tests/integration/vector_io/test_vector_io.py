# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.vector_io import Chunk


@pytest.fixture(scope="session")
def sample_chunks():
    return [
        Chunk(
            content="Python is a high-level programming language that emphasizes code readability and allows programmers to express concepts in fewer lines of code than would be possible in languages such as C++ or Java.",
            metadata={"document_id": "doc1"},
        ),
        Chunk(
            content="Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience without being explicitly programmed, using statistical techniques to give computer systems the ability to progressively improve performance on a specific task.",
            metadata={"document_id": "doc2"},
        ),
        Chunk(
            content="Data structures are fundamental to computer science because they provide organized ways to store and access data efficiently, enable faster processing of data through optimized algorithms, and form the building blocks for more complex software systems.",
            metadata={"document_id": "doc3"},
        ),
        Chunk(
            content="Neural networks are inspired by biological neural networks found in animal brains, using interconnected nodes called artificial neurons to process information through weighted connections that can be trained to recognize patterns and solve complex problems through iterative learning.",
            metadata={"document_id": "doc4"},
        ),
    ]


@pytest.fixture(scope="function")
def client_with_empty_registry(client_with_models):
    def clear_registry():
        vector_dbs = [vector_db.identifier for vector_db in client_with_models.vector_dbs.list()]
        for vector_db_id in vector_dbs:
            client_with_models.vector_dbs.unregister(vector_db_id=vector_db_id)

    clear_registry()
    yield client_with_models

    # you must clean after the last test if you were running tests against
    # a stateful server instance
    clear_registry()


def test_vector_db_retrieve(client_with_empty_registry, embedding_model_id):
    # Register a memory bank first
    vector_db_id = "test_vector_db"
    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=384,
    )

    # Retrieve the memory bank and validate its properties
    response = client_with_empty_registry.vector_dbs.retrieve(vector_db_id=vector_db_id)
    assert response is not None
    assert response.identifier == vector_db_id
    assert response.embedding_model == embedding_model_id
    assert response.provider_resource_id == vector_db_id


def test_vector_db_register(client_with_empty_registry, embedding_model_id):
    vector_db_id = "test_vector_db"
    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=384,
    )

    vector_dbs_after_register = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    assert vector_dbs_after_register == [vector_db_id]

    client_with_empty_registry.vector_dbs.unregister(vector_db_id=vector_db_id)

    vector_dbs = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    assert len(vector_dbs) == 0


@pytest.mark.parametrize(
    "test_case",
    [
        ("What makes Python different from C++ and Java?", "doc1"),
        ("How do systems learn without explicit programming?", "doc2"),
        ("Why are data structures important in computer science?", "doc3"),
        ("What is the biological inspiration for neural networks?", "doc4"),
        ("How does machine learning improve over time?", "doc2"),
    ],
)
def test_insert_chunks(client_with_empty_registry, embedding_model_id, sample_chunks, test_case):
    vector_db_id = "test_vector_db"
    client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model_id,
        embedding_dimension=384,
    )

    client_with_empty_registry.vector_io.insert(
        vector_db_id=vector_db_id,
        chunks=sample_chunks,
    )

    response = client_with_empty_registry.vector_io.query(
        vector_db_id=vector_db_id,
        query="What is the capital of France?",
    )
    assert response is not None
    assert len(response.chunks) > 1
    assert len(response.scores) > 1

    query, expected_doc_id = test_case
    response = client_with_empty_registry.vector_io.query(
        vector_db_id=vector_db_id,
        query=query,
    )
    assert response is not None
    top_match = response.chunks[0]
    assert top_match is not None
    assert top_match.metadata["document_id"] == expected_doc_id, f"Query '{query}' should match {expected_doc_id}"
