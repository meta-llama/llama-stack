# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack.apis.tools import RAGDocument, RAGQueryResult, ToolInvocationResult
from llama_stack.providers.datatypes import Api


@pytest.fixture
def sample_search_query():
    return "What are the latest developments in quantum computing?"


@pytest.fixture
def sample_wolfram_alpha_query():
    return "What is the square root of 16?"


@pytest.fixture
def sample_documents():
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]
    return [
        RAGDocument(
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={},
        )
        for i, url in enumerate(urls)
    ]


class TestTools:
    @pytest.mark.asyncio
    async def test_web_search_tool(self, tools_stack, sample_search_query):
        """Test the web search tool functionality."""
        if "TAVILY_SEARCH_API_KEY" not in os.environ:
            pytest.skip("TAVILY_SEARCH_API_KEY not set, skipping test")

        tools_impl = tools_stack.impls[Api.tool_runtime]

        # Execute the tool
        response = await tools_impl.invoke_tool(tool_name="web_search", kwargs={"query": sample_search_query})

        # Verify the response
        assert isinstance(response, ToolInvocationResult)
        assert response.content is not None
        assert len(response.content) > 0
        assert isinstance(response.content, str)

    @pytest.mark.asyncio
    async def test_wolfram_alpha_tool(self, tools_stack, sample_wolfram_alpha_query):
        """Test the wolfram alpha tool functionality."""
        if "WOLFRAM_ALPHA_API_KEY" not in os.environ:
            pytest.skip("WOLFRAM_ALPHA_API_KEY not set, skipping test")

        tools_impl = tools_stack.impls[Api.tool_runtime]

        response = await tools_impl.invoke_tool(tool_name="wolfram_alpha", kwargs={"query": sample_wolfram_alpha_query})

        # Verify the response
        assert isinstance(response, ToolInvocationResult)
        assert response.content is not None
        assert len(response.content) > 0
        assert isinstance(response.content, str)

    @pytest.mark.asyncio
    async def test_rag_tool(self, tools_stack, sample_documents):
        """Test the memory tool functionality."""
        vector_dbs_impl = tools_stack.impls[Api.vector_dbs]
        tools_impl = tools_stack.impls[Api.tool_runtime]

        # Register memory bank
        await vector_dbs_impl.register_vector_db(
            vector_db_id="test_bank",
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",
        )

        # Insert documents into memory
        await tools_impl.rag_tool.insert(
            documents=sample_documents,
            vector_db_id="test_bank",
            chunk_size_in_tokens=512,
        )

        # Execute the memory tool
        response = await tools_impl.rag_tool.query(
            content="What are the main topics covered in the documentation?",
            vector_db_ids=["test_bank"],
        )

        # Verify the response
        assert isinstance(response, RAGQueryResult)
        assert response.content is not None
        assert len(response.content) > 0
