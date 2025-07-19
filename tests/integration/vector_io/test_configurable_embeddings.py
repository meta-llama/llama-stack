# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client import BadRequestError  # type: ignore

# Local copy of the sample_chunks fixture so this module can run independently
from llama_stack.apis.vector_io import Chunk


# Re-use the same sample data defined in test_openai_vector_stores.py
@pytest.fixture(scope="session")
def sample_chunks():
    return [
        Chunk(
            content="Python is a high-level programming language that emphasizes code readability.",
            metadata={"document_id": "doc1", "topic": "programming"},
        ),
        Chunk(
            content="Machine learning is a subset of artificial intelligence that enables systems to automatically learn and improve from experience.",
            metadata={"document_id": "doc2", "topic": "ai"},
        ),
        Chunk(
            content="Data structures are fundamental to computer science because they provide organized ways to store and access data efficiently.",
            metadata={"document_id": "doc3", "topic": "computer_science"},
        ),
        Chunk(
            content="Neural networks are inspired by biological neural networks found in animal brains, using interconnected nodes called artificial neurons to process information.",
            metadata={"document_id": "doc4", "topic": "ai"},
        ),
    ]


# Re-use the existing integration fixtures:
#   * `client_with_models` - a LlamaStackClient with providers & models registered
#   * `sample_chunks` - four small text chunks defined in test_openai_vector_stores.py


@pytest.mark.integration
def test_vector_db_custom_embedding_happy_path(client_with_models, sample_chunks):
    """Register a vector DB with an explicit, non-default embedding model and verify e2e search works."""
    vector_db_id = "embed_happy_test"

    # Register the vector DB with a custom embedding model.  The model already exists
    # in the default run-config so we don't need to register it separately.
    client_with_models.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model="all-MiniLM-L6-v2",  # explicit even though it is default
        embedding_dimension=384,
    )

    try:
        # Insert one chunk so the DB is not empty
        client_with_models.vector_io.insert(vector_db_id=vector_db_id, chunks=sample_chunks[:1])

        # Simple similarity query - should return the chunk we just inserted
        res = client_with_models.vector_io.query(vector_db_id=vector_db_id, query="programming language")

        assert res is not None
        assert len(res.chunks) > 0, "Expected at least one chunk to be retrieved"
        assert res.scores[0] > 0, "Top score should be positive"
    finally:
        # Always clean up so later tests start fresh
        client_with_models.vector_dbs.unregister(vector_db_id=vector_db_id)


@pytest.mark.integration
def test_vector_db_register_rejects_non_embedding_model(client_with_models):
    """Attempting to register a vector DB with a non-embedding model should raise an error."""
    vector_db_id = "embed_failure_test"

    with pytest.raises((BadRequestError, ValueError)):
        client_with_models.vector_dbs.register(
            vector_db_id=vector_db_id,
            # This is an LLM, *not* an embedding model ⇒ should fail
            embedding_model="meta-llama/Llama-3.1-8B-Instruct",
            embedding_dimension=4096,
        )
