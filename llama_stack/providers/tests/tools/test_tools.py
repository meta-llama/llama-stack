# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack.apis.inference import UserMessage
from llama_stack.apis.memory import MemoryBankDocument
from llama_stack.apis.memory_banks import VectorMemoryBankParams
from llama_stack.apis.tools import ToolInvocationResult
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
        "datasets.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]
    return [
        MemoryBankDocument(
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
        response = await tools_impl.invoke_tool(
            tool_name="web_search", args={"query": sample_search_query}
        )

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

        response = await tools_impl.invoke_tool(
            tool_name="wolfram_alpha", args={"query": sample_wolfram_alpha_query}
        )

        # Verify the response
        assert isinstance(response, ToolInvocationResult)
        assert response.content is not None
        assert len(response.content) > 0
        assert isinstance(response.content, str)

    @pytest.mark.asyncio
    async def test_memory_tool(self, tools_stack, sample_documents):
        """Test the memory tool functionality."""
        memory_banks_impl = tools_stack.impls[Api.memory_banks]
        memory_impl = tools_stack.impls[Api.memory]
        tools_impl = tools_stack.impls[Api.tool_runtime]

        # Register memory bank
        await memory_banks_impl.register_memory_bank(
            memory_bank_id="test_bank",
            params=VectorMemoryBankParams(
                embedding_model="all-MiniLM-L6-v2",
                chunk_size_in_tokens=512,
                overlap_size_in_tokens=64,
            ),
            provider_id="faiss",
        )

        # Insert documents into memory
        await memory_impl.insert_documents(
            bank_id="test_bank",
            documents=sample_documents,
        )

        # Execute the memory tool
        response = await tools_impl.invoke_tool(
            tool_name="memory",
            args={
                "messages": [
                    UserMessage(
                        content="What are the main topics covered in the documentation?",
                    )
                ],
                "memory_bank_ids": ["test_bank"],
            },
        )

        # Verify the response
        assert isinstance(response, ToolInvocationResult)
        assert response.content is not None
        assert len(response.content) > 0
