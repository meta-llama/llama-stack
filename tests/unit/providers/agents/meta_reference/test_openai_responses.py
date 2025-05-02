# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock

import pytest

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseOutputMessage,
)
from llama_stack.apis.inference.inference import (
    OpenAIUserMessageParam,
)
from llama_stack.apis.tools.tools import Tool, ToolGroups, ToolInvocationResult, ToolParameter, ToolRuntime
from llama_stack.providers.inline.agents.meta_reference.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.utils.kvstore import KVStore
from tests.unit.providers.agents.meta_reference.fixtures import load_chat_completion_fixture


@pytest.fixture
def mock_kvstore():
    kvstore = AsyncMock(spec=KVStore)
    return kvstore


@pytest.fixture
def mock_inference_api():
    inference_api = AsyncMock()
    return inference_api


@pytest.fixture
def mock_tool_groups_api():
    tool_groups_api = AsyncMock(spec=ToolGroups)
    return tool_groups_api


@pytest.fixture
def mock_tool_runtime_api():
    tool_runtime_api = AsyncMock(spec=ToolRuntime)
    return tool_runtime_api


@pytest.fixture
def openai_responses_impl(mock_kvstore, mock_inference_api, mock_tool_groups_api, mock_tool_runtime_api):
    return OpenAIResponsesImpl(
        persistence_store=mock_kvstore,
        inference_api=mock_inference_api,
        tool_groups_api=mock_tool_groups_api,
        tool_runtime_api=mock_tool_runtime_api,
    )


@pytest.mark.asyncio
async def test_create_openai_response_with_string_input(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a simple string input."""
    # Setup
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Load the chat completion fixture
    mock_chat_completion = load_chat_completion_fixture("simple_chat_completion.yaml")
    mock_inference_api.openai_chat_completion.return_value = mock_chat_completion

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        temperature=0.1,
    )

    # Verify
    mock_inference_api.openai_chat_completion.assert_called_once_with(
        model=model,
        messages=[OpenAIUserMessageParam(role="user", content="What is the capital of Ireland?", name=None)],
        tools=None,
        stream=False,
        temperature=0.1,
    )
    openai_responses_impl.persistence_store.set.assert_called_once()
    assert result.model == model
    assert len(result.output) == 1
    assert isinstance(result.output[0], OpenAIResponseOutputMessage)
    assert result.output[0].content[0].text == "Dublin"


@pytest.mark.asyncio
async def test_create_openai_response_with_string_input_with_tools(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a simple string input and tools."""
    # Setup
    input_text = "What is the capital of Ireland?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Load the chat completion fixtures
    tool_call_completion = load_chat_completion_fixture("tool_call_completion.yaml")
    tool_response_completion = load_chat_completion_fixture("simple_chat_completion.yaml")

    mock_inference_api.openai_chat_completion.side_effect = [
        tool_call_completion,
        tool_response_completion,
    ]

    openai_responses_impl.tool_groups_api.get_tool.return_value = Tool(
        identifier="web_search",
        provider_id="client",
        toolgroup_id="web_search",
        tool_host="client",
        description="Search the web for information",
        parameters=[
            ToolParameter(name="query", parameter_type="string", description="The query to search for", required=True)
        ],
    )

    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Dublin",
    )

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        temperature=0.1,
        tools=[
            OpenAIResponseInputToolWebSearch(
                name="web_search",
            )
        ],
    )

    # Verify
    first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
    assert first_call.kwargs["messages"][0].content == "What is the capital of Ireland?"
    assert first_call.kwargs["tools"] is not None
    assert first_call.kwargs["temperature"] == 0.1

    second_call = mock_inference_api.openai_chat_completion.call_args_list[1]
    assert second_call.kwargs["messages"][-1].content == "Dublin"
    assert second_call.kwargs["temperature"] == 0.1

    openai_responses_impl.tool_groups_api.get_tool.assert_called_once_with("web_search")
    openai_responses_impl.tool_runtime_api.invoke_tool.assert_called_once_with(
        tool_name="web_search",
        kwargs={"query": "What is the capital of Ireland?"},
    )

    openai_responses_impl.persistence_store.set.assert_called_once()

    # Check that we got the content from our mocked tool execution result
    assert len(result.output) >= 1
    assert isinstance(result.output[1], OpenAIResponseOutputMessage)
    assert result.output[1].content[0].text == "Dublin"
