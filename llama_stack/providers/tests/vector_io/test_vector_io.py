# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

import pytest

from llama_stack.apis.tools import RAGDocument
from llama_stack.apis.vector_dbs import ListVectorDBsResponse, VectorDB
from llama_stack.apis.vector_io import QueryChunksResponse
from llama_stack.providers.utils.memory.vector_store import make_overlapped_chunks

# How to run this test:
#
# pytest llama_stack/providers/tests/vector_io/test_vector_io.py \
#   -m "pgvector" --env EMBEDDING_DIMENSION=384 PGVECTOR_PORT=7432 \
#   -v -s --tb=short --disable-warnings


@pytest.fixture(scope="session")
def sample_chunks():
    docs = [
        RAGDocument(
            document_id="doc1",
            content="Python is a high-level programming language.",
            metadata={"category": "programming", "difficulty": "beginner"},
        ),
        RAGDocument(
            document_id="doc2",
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "AI", "difficulty": "advanced"},
        ),
        RAGDocument(
            document_id="doc3",
            content="Data structures are fundamental to computer science.",
            metadata={"category": "computer science", "difficulty": "intermediate"},
        ),
        RAGDocument(
            document_id="doc4",
            content="Neural networks are inspired by biological neural networks.",
            metadata={"category": "AI", "difficulty": "advanced"},
        ),
    ]
    chunks = []
    for doc in docs:
        chunks.extend(make_overlapped_chunks(doc.document_id, doc.content, window_len=512, overlap_len=64))
    return chunks


async def register_vector_db(vector_dbs_impl: VectorDB, embedding_model: str):
    vector_db_id = f"test_vector_db_{uuid.uuid4().hex}"
    return await vector_dbs_impl.register_vector_db(
        vector_db_id=vector_db_id,
        embedding_model=embedding_model,
        embedding_dimension=384,
    )


class TestVectorIO:
    @pytest.mark.asyncio
    async def test_banks_list(self, vector_io_stack, embedding_model):
        _, vector_dbs_impl = vector_io_stack

        # Register a test bank
        registered_vector_db = await register_vector_db(vector_dbs_impl, embedding_model)

        try:
            # Verify our bank shows up in list
            response = await vector_dbs_impl.list_vector_dbs()
            assert isinstance(response, ListVectorDBsResponse)
            assert any(vector_db.vector_db_id == registered_vector_db.vector_db_id for vector_db in response.data)
        finally:
            # Clean up
            await vector_dbs_impl.unregister_vector_db(registered_vector_db.vector_db_id)

        # Verify our bank was removed
        response = await vector_dbs_impl.list_vector_dbs()
        assert isinstance(response, ListVectorDBsResponse)
        assert all(vector_db.vector_db_id != registered_vector_db.vector_db_id for vector_db in response.data)

    @pytest.mark.asyncio
    async def test_banks_register(self, vector_io_stack, embedding_model):
        _, vector_dbs_impl = vector_io_stack

        vector_db_id = f"test_vector_db_{uuid.uuid4().hex}"

        try:
            # Register initial bank
            await vector_dbs_impl.register_vector_db(
                vector_db_id=vector_db_id,
                embedding_model=embedding_model,
                embedding_dimension=384,
            )

            # Verify our bank exists
            response = await vector_dbs_impl.list_vector_dbs()
            assert isinstance(response, ListVectorDBsResponse)
            assert any(vector_db.vector_db_id == vector_db_id for vector_db in response.data)

            # Try registering same bank again
            await vector_dbs_impl.register_vector_db(
                vector_db_id=vector_db_id,
                embedding_model=embedding_model,
                embedding_dimension=384,
            )

            # Verify still only one instance of our bank
            response = await vector_dbs_impl.list_vector_dbs()
            assert isinstance(response, ListVectorDBsResponse)
            assert len([vector_db for vector_db in response.data if vector_db.vector_db_id == vector_db_id]) == 1
        finally:
            # Clean up
            await vector_dbs_impl.unregister_vector_db(vector_db_id)

    @pytest.mark.asyncio
    async def test_query_documents(self, vector_io_stack, embedding_model, sample_chunks):
        vector_io_impl, vector_dbs_impl = vector_io_stack

        with pytest.raises(ValueError):
            await vector_io_impl.insert_chunks("test_vector_db", sample_chunks)

        registered_db = await register_vector_db(vector_dbs_impl, embedding_model)
        await vector_io_impl.insert_chunks(registered_db.vector_db_id, sample_chunks)

        query1 = "programming language"
        response1 = await vector_io_impl.query_chunks(registered_db.vector_db_id, query1)
        assert_valid_response(response1)
        assert any("Python" in chunk.content for chunk in response1.chunks)

        # Test case 3: Query with semantic similarity
        query3 = "AI and brain-inspired computing"
        response3 = await vector_io_impl.query_chunks(registered_db.vector_db_id, query3)
        assert_valid_response(response3)
        assert any("neural networks" in chunk.content.lower() for chunk in response3.chunks)

        # Test case 4: Query with limit on number of results
        query4 = "computer"
        params4 = {"max_chunks": 2}
        response4 = await vector_io_impl.query_chunks(registered_db.vector_db_id, query4, params4)
        assert_valid_response(response4)
        assert len(response4.chunks) <= 2

        # Test case 5: Query with threshold on similarity score
        query5 = "quantum computing"  # Not directly related to any document
        params5 = {"score_threshold": 0.01}
        response5 = await vector_io_impl.query_chunks(registered_db.vector_db_id, query5, params5)
        assert_valid_response(response5)
        print("The scores are:", response5.scores)
        assert all(score >= 0.01 for score in response5.scores)


def assert_valid_response(response: QueryChunksResponse):
    assert len(response.chunks) > 0
    assert len(response.scores) > 0
    assert len(response.chunks) == len(response.scores)
    for chunk in response.chunks:
        assert isinstance(chunk.content, str)
