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
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIUserMessageParam,
)
from llama_stack.apis.tools.tools import Tool, ToolGroups, ToolInvocationResult, ToolParameter, ToolRuntime
from llama_stack.providers.inline.agents.meta_reference.openai_responses import (
    OpenAIResponsesImpl,
)
from llama_stack.providers.utils.kvstore import KVStore


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
    input_text = "Hello, world!"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_chat_completion = OpenAIChatCompletion(
        id="chat-completion-123",
        choices=[
            OpenAIChoice(
                message=OpenAIAssistantMessageParam(content="Hello! How can I help you?"),
                finish_reason="stop",
                index=0,
            )
        ],
        created=1234567890,
        model=model,
    )
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
        messages=[OpenAIUserMessageParam(role="user", content="Hello, world!", name=None)],
        tools=None,
        stream=False,
        temperature=0.1,
    )
    openai_responses_impl.persistence_store.set.assert_called_once()
    assert result.model == model
    assert len(result.output) == 1
    assert isinstance(result.output[0], OpenAIResponseOutputMessage)
    assert result.output[0].content[0].text == "Hello! How can I help you?"


@pytest.mark.asyncio
async def test_create_openai_response_with_string_input_with_tools(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a simple string input and tools."""
    # Setup
    input_text = "What was the score of todays game?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_chat_completions = [
        OpenAIChatCompletion(
            id="chat-completion-123",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(
                        tool_calls=[
                            OpenAIChatCompletionToolCall(
                                id="tool_call_123",
                                type="function",
                                function=OpenAIChatCompletionToolCallFunction(
                                    name="web_search", arguments='{"query":"What was the score of todays game?"}'
                                ),
                            )
                        ],
                    ),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1234567890,
            model=model,
        ),
        OpenAIChatCompletion(
            id="chat-completion-123",
            choices=[
                OpenAIChoice(
                    message=OpenAIAssistantMessageParam(content="The score of todays game was 10-12"),
                    finish_reason="stop",
                    index=0,
                )
            ],
            created=1234567890,
            model=model,
        ),
    ]

    mock_inference_api.openai_chat_completion.side_effect = mock_chat_completions

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
        content="The score of todays game was 10-12",
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
    assert first_call.kwargs["messages"][0].content == "What was the score of todays game?"
    assert first_call.kwargs["tools"] is not None
    assert first_call.kwargs["temperature"] == 0.1

    second_call = mock_inference_api.openai_chat_completion.call_args_list[1]
    assert second_call.kwargs["messages"][-1].content == "The score of todays game was 10-12"
    assert second_call.kwargs["temperature"] == 0.1

    openai_responses_impl.tool_groups_api.get_tool.assert_called_once_with("web_search")
    openai_responses_impl.tool_runtime_api.invoke_tool.assert_called_once_with(
        tool_name="web_search",
        kwargs={"query": "What was the score of todays game?"},
    )

    openai_responses_impl.persistence_store.set.assert_called_once()

    # Check that we got the content from our mocked tool execution result
    assert len(result.output) >= 1
    assert isinstance(result.output[1], OpenAIResponseOutputMessage)
    assert result.output[1].content[0].text == "The score of todays game was 10-12"
