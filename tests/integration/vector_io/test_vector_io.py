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


def test_vector_db_retrieve(client_with_empty_registry, embedding_model_id, embedding_dimension):
    vector_db_name = "test_vector_db"
    register_response = client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_name,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    actual_vector_db_id = register_response.identifier

    # Retrieve the memory bank and validate its properties
    response = client_with_empty_registry.vector_dbs.retrieve(vector_db_id=actual_vector_db_id)
    assert response is not None
    assert response.identifier == actual_vector_db_id
    assert response.embedding_model == embedding_model_id
    assert response.identifier.startswith("vs_")


def test_vector_db_register(client_with_empty_registry, embedding_model_id, embedding_dimension):
    vector_db_name = "test_vector_db"
    response = client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_name,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    actual_vector_db_id = response.identifier
    assert actual_vector_db_id.startswith("vs_")
    assert actual_vector_db_id != vector_db_name

    vector_dbs_after_register = [vector_db.identifier for vector_db in client_with_empty_registry.vector_dbs.list()]
    assert vector_dbs_after_register == [actual_vector_db_id]

    vector_stores = client_with_empty_registry.vector_stores.list()
    assert len(vector_stores.data) == 1
    vector_store = vector_stores.data[0]
    assert vector_store.id == actual_vector_db_id
    assert vector_store.name == vector_db_name

    client_with_empty_registry.vector_dbs.unregister(vector_db_id=actual_vector_db_id)

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
def test_insert_chunks(client_with_empty_registry, embedding_model_id, embedding_dimension, sample_chunks, test_case):
    vector_db_name = "test_vector_db"
    register_response = client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_name,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    actual_vector_db_id = register_response.identifier

    client_with_empty_registry.vector_io.insert(
        vector_db_id=actual_vector_db_id,
        chunks=sample_chunks,
    )

    response = client_with_empty_registry.vector_io.query(
        vector_db_id=actual_vector_db_id,
        query="What is the capital of France?",
    )
    assert response is not None
    assert len(response.chunks) > 1
    assert len(response.scores) > 1

    query, expected_doc_id = test_case
    response = client_with_empty_registry.vector_io.query(
        vector_db_id=actual_vector_db_id,
        query=query,
    )
    assert response is not None
    top_match = response.chunks[0]
    assert top_match is not None
    assert top_match.metadata["document_id"] == expected_doc_id, f"Query '{query}' should match {expected_doc_id}"


def test_insert_chunks_with_precomputed_embeddings(client_with_empty_registry, embedding_model_id, embedding_dimension):
    vector_io_provider_params_dict = {
        "inline::milvus": {"score_threshold": -1.0},
        "remote::qdrant": {"score_threshold": -1.0},
        "inline::qdrant": {"score_threshold": -1.0},
    }
    vector_db_name = "test_precomputed_embeddings_db"
    register_response = client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_name,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    actual_vector_db_id = register_response.identifier

    chunks_with_embeddings = [
        Chunk(
            content="This is a test chunk with precomputed embedding.",
            metadata={"document_id": "doc1", "source": "precomputed", "chunk_id": "chunk1"},
            embedding=[0.1] * int(embedding_dimension),
        ),
    ]

    client_with_empty_registry.vector_io.insert(
        vector_db_id=actual_vector_db_id,
        chunks=chunks_with_embeddings,
    )

    provider = [p.provider_id for p in client_with_empty_registry.providers.list() if p.api == "vector_io"][0]
    response = client_with_empty_registry.vector_io.query(
        vector_db_id=actual_vector_db_id,
        query="precomputed embedding test",
        params=vector_io_provider_params_dict.get(provider, None),
    )

    # Verify the top result is the expected document
    assert response is not None
    assert len(response.chunks) > 0, (
        f"provider params for {provider} = {vector_io_provider_params_dict.get(provider, None)}"
    )
    assert response.chunks[0].metadata["document_id"] == "doc1"
    assert response.chunks[0].metadata["source"] == "precomputed"


# expect this test to fail
def test_query_returns_valid_object_when_identical_to_embedding_in_vdb(
    client_with_empty_registry, embedding_model_id, embedding_dimension
):
    vector_io_provider_params_dict = {
        "inline::milvus": {"score_threshold": 0.0},
        "remote::qdrant": {"score_threshold": 0.0},
        "inline::qdrant": {"score_threshold": 0.0},
    }
    vector_db_name = "test_precomputed_embeddings_db"
    register_response = client_with_empty_registry.vector_dbs.register(
        vector_db_id=vector_db_name,
        embedding_model=embedding_model_id,
        embedding_dimension=embedding_dimension,
    )

    actual_vector_db_id = register_response.identifier

    chunks_with_embeddings = [
        Chunk(
            content="duplicate",
            metadata={"document_id": "doc1", "source": "precomputed"},
            embedding=[0.1] * int(embedding_dimension),
        ),
    ]

    client_with_empty_registry.vector_io.insert(
        vector_db_id=actual_vector_db_id,
        chunks=chunks_with_embeddings,
    )

    provider = [p.provider_id for p in client_with_empty_registry.providers.list() if p.api == "vector_io"][0]
    response = client_with_empty_registry.vector_io.query(
        vector_db_id=actual_vector_db_id,
        query="duplicate",
        params=vector_io_provider_params_dict.get(provider, None),
    )

    # Verify the top result is the expected document
    assert response is not None
    assert len(response.chunks) > 0
    assert response.chunks[0].metadata["document_id"] == "doc1"
    assert response.chunks[0].metadata["source"] == "precomputed"
