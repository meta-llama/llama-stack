# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import List

import pytest
import requests
from pydantic import TypeAdapter

from llama_stack.apis.tools import (
    DefaultRAGQueryGeneratorConfig,
    RAGDocument,
    RAGQueryConfig,
    RAGQueryResult,
)
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.providers.utils.memory.vector_store import interleaved_content_as_str


class TestRAGToolEndpoints:
    @pytest.fixture
    def base_url(self) -> str:
        return "http://localhost:8321/v1"  # Adjust port if needed

    @pytest.fixture
    def sample_documents(self) -> List[RAGDocument]:
        return [
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
        ]

    @pytest.mark.asyncio
    async def test_rag_workflow(self, base_url: str, sample_documents: List[RAGDocument]):
        vector_db_payload = {
            "vector_db_id": "test_vector_db",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
        }

        response = requests.post(f"{base_url}/vector-dbs", json=vector_db_payload)
        assert response.status_code == 200
        vector_db = VectorDB(**response.json())

        insert_payload = {
            "documents": [json.loads(doc.model_dump_json()) for doc in sample_documents],
            "vector_db_id": vector_db.identifier,
            "chunk_size_in_tokens": 512,
        }

        response = requests.post(
            f"{base_url}/tool-runtime/rag-tool/insert-documents",
            json=insert_payload,
        )
        assert response.status_code == 200

        query = "What is Python?"
        query_config = RAGQueryConfig(
            query_generator_config=DefaultRAGQueryGeneratorConfig(),
            max_tokens_in_context=4096,
            max_chunks=2,
        )

        query_payload = {
            "content": query,
            "query_config": json.loads(query_config.model_dump_json()),
            "vector_db_ids": [vector_db.identifier],
        }

        response = requests.post(
            f"{base_url}/tool-runtime/rag-tool/query-context",
            json=query_payload,
        )
        assert response.status_code == 200
        result = response.json()
        result = TypeAdapter(RAGQueryResult).validate_python(result)

        content_str = interleaved_content_as_str(result.content)
        print(f"content: {content_str}")
        assert len(content_str) > 0
        assert "Python" in content_str

        # Clean up: Delete the vector DB
        response = requests.delete(f"{base_url}/vector-dbs/{vector_db.identifier}")
        assert response.status_code == 200
