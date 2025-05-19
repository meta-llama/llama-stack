# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from unittest.mock import AsyncMock, patch

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInputItemList,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputToolFunction,
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageWebSearchToolCall,
)
from llama_stack.apis.inference.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIDeveloperMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.apis.tools.tools import Tool, ToolGroups, ToolInvocationResult, ToolParameter, ToolRuntime
from llama_stack.providers.inline.agents.meta_reference.openai_responses import (
    OpenAIResponsePreviousResponseWithInputItems,
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
    assert isinstance(result.output[0], OpenAIResponseMessage)
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
    assert isinstance(result.output[1], OpenAIResponseMessage)
    assert result.output[1].content[0].text == "Dublin"


@pytest.mark.asyncio
async def test_create_openai_response_with_tool_call_type_none(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with a tool call response that has a type of None."""
    # Setup
    input_text = "How hot it is in San Francisco today?"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    async def fake_stream():
        yield ChatCompletionChunk(
            id="123",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tc_123",
                                function=ChoiceDeltaToolCallFunction(name="get_weather", arguments="{}"),
                                type=None,
                            )
                        ]
                    ),
                ),
            ],
            created=1,
            model=model,
            object="chat.completion.chunk",
        )

    mock_inference_api.openai_chat_completion.return_value = fake_stream()

    # Execute
    result = await openai_responses_impl.create_openai_response(
        input=input_text,
        model=model,
        stream=True,
        temperature=0.1,
        tools=[
            OpenAIResponseInputToolFunction(
                name="get_weather",
                description="Get current temperature for a given location.",
                parameters={
                    "location": "string",
                },
            )
        ],
    )

    # Verify
    first_call = mock_inference_api.openai_chat_completion.call_args_list[0]
    assert first_call.kwargs["messages"][0].content == input_text
    assert first_call.kwargs["tools"] is not None
    assert first_call.kwargs["temperature"] == 0.1

    # Check that we got the content from our mocked tool execution result
    chunks = [chunk async for chunk in result]
    assert len(chunks) > 0
    assert chunks[0].response.output[0].type == "function_call"
    assert chunks[0].response.output[0].name == "get_weather"


@pytest.mark.asyncio
async def test_create_openai_response_with_multiple_messages(openai_responses_impl, mock_inference_api):
    """Test creating an OpenAI response with multiple messages."""
    # Setup
    input_messages = [
        OpenAIResponseMessage(role="developer", content="You are a helpful assistant", name=None),
        OpenAIResponseMessage(role="user", content="Name some towns in Ireland", name=None),
        OpenAIResponseMessage(
            role="assistant",
            content=[
                OpenAIResponseInputMessageContentText(text="Galway, Longford, Sligo"),
                OpenAIResponseInputMessageContentText(text="Dublin"),
            ],
            name=None,
        ),
        OpenAIResponseMessage(role="user", content="Which is the largest town in Ireland?", name=None),
    ]
    model = "meta-llama/Llama-3.1-8B-Instruct"

    mock_inference_api.openai_chat_completion.return_value = load_chat_completion_fixture("simple_chat_completion.yaml")

    # Execute
    await openai_responses_impl.create_openai_response(
        input=input_messages,
        model=model,
        temperature=0.1,
    )

    # Verify the the correct messages were sent to the inference API i.e.
    # All of the responses message were convered to the chat completion message objects
    inference_messages = mock_inference_api.openai_chat_completion.call_args_list[0].kwargs["messages"]
    for i, m in enumerate(input_messages):
        if isinstance(m.content, str):
            assert inference_messages[i].content == m.content
        else:
            assert inference_messages[i].content[0].text == m.content[0].text
            assert isinstance(inference_messages[i].content[0], OpenAIChatCompletionContentPartTextParam)
        assert inference_messages[i].role == m.role
        if m.role == "user":
            assert isinstance(inference_messages[i], OpenAIUserMessageParam)
        elif m.role == "assistant":
            assert isinstance(inference_messages[i], OpenAIAssistantMessageParam)
        else:
            assert isinstance(inference_messages[i], OpenAIDeveloperMessageParam)


@pytest.mark.asyncio
async def test_prepend_previous_response_none(openai_responses_impl):
    """Test prepending no previous response to a new response."""

    input = await openai_responses_impl._prepend_previous_response("fake_input", None)
    assert input == "fake_input"


@pytest.mark.asyncio
@patch.object(OpenAIResponsesImpl, "_get_previous_response_with_input")
async def test_prepend_previous_response_basic(get_previous_response_with_input, openai_responses_impl):
    """Test prepending a basic previous response to a new response."""

    input_item_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseInputMessageContentText(text="fake_previous_input")],
        role="user",
    )
    input_items = OpenAIResponseInputItemList(data=[input_item_message])
    response_output_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseOutputMessageContentOutputText(text="fake_response")],
        status="completed",
        role="assistant",
    )
    response = OpenAIResponseObject(
        created_at=1,
        id="resp_123",
        model="fake_model",
        output=[response_output_message],
        status="completed",
    )
    previous_response = OpenAIResponsePreviousResponseWithInputItems(
        input_items=input_items,
        response=response,
    )
    get_previous_response_with_input.return_value = previous_response

    input = await openai_responses_impl._prepend_previous_response("fake_input", "resp_123")

    assert len(input) == 3
    # Check for previous input
    assert isinstance(input[0], OpenAIResponseMessage)
    assert input[0].content[0].text == "fake_previous_input"
    # Check for previous output
    assert isinstance(input[1], OpenAIResponseMessage)
    assert input[1].content[0].text == "fake_response"
    # Check for new input
    assert isinstance(input[2], OpenAIResponseMessage)
    assert input[2].content == "fake_input"


@pytest.mark.asyncio
@patch.object(OpenAIResponsesImpl, "_get_previous_response_with_input")
async def test_prepend_previous_response_web_search(get_previous_response_with_input, openai_responses_impl):
    """Test prepending a web search previous response to a new response."""

    input_item_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseInputMessageContentText(text="fake_previous_input")],
        role="user",
    )
    input_items = OpenAIResponseInputItemList(data=[input_item_message])
    output_web_search = OpenAIResponseOutputMessageWebSearchToolCall(
        id="ws_123",
        status="completed",
    )
    output_message = OpenAIResponseMessage(
        id="123",
        content=[OpenAIResponseOutputMessageContentOutputText(text="fake_web_search_response")],
        status="completed",
        role="assistant",
    )
    response = OpenAIResponseObject(
        created_at=1,
        id="resp_123",
        model="fake_model",
        output=[output_web_search, output_message],
        status="completed",
    )
    previous_response = OpenAIResponsePreviousResponseWithInputItems(
        input_items=input_items,
        response=response,
    )
    get_previous_response_with_input.return_value = previous_response

    input_messages = [OpenAIResponseMessage(content="fake_input", role="user")]
    input = await openai_responses_impl._prepend_previous_response(input_messages, "resp_123")

    assert len(input) == 4
    # Check for previous input
    assert isinstance(input[0], OpenAIResponseMessage)
    assert input[0].content[0].text == "fake_previous_input"
    # Check for previous output web search tool call
    assert isinstance(input[1], OpenAIResponseOutputMessageWebSearchToolCall)
    # Check for previous output web search response
    assert isinstance(input[2], OpenAIResponseMessage)
    assert input[2].content[0].text == "fake_web_search_response"
    # Check for new input
    assert isinstance(input[3], OpenAIResponseMessage)
    assert input[3].content == "fake_input"
