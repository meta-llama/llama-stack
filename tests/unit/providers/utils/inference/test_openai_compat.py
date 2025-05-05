# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.common.content_types import TextContentItem
from llama_stack.apis.inference.inference import (
    CompletionMessage,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAISystemMessageParam,
    OpenAIUserMessageParam,
    SystemMessage,
    UserMessage,
)
from llama_stack.models.llama.datatypes import BuiltinTool, StopReason, ToolCall
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    openai_messages_to_messages,
)


@pytest.mark.asyncio
async def test_convert_message_to_openai_dict():
    message = UserMessage(content=[TextContentItem(text="Hello, world!")], role="user")
    assert await convert_message_to_openai_dict(message) == {
        "role": "user",
        "content": [{"type": "text", "text": "Hello, world!"}],
    }


# Test convert_message_to_openai_dict with a tool call
@pytest.mark.asyncio
async def test_convert_message_to_openai_dict_with_tool_call():
    message = CompletionMessage(
        content="",
        tool_calls=[
            ToolCall(call_id="123", tool_name="test_tool", arguments_json='{"foo": "bar"}', arguments={"foo": "bar"})
        ],
        stop_reason=StopReason.end_of_turn,
    )

    openai_dict = await convert_message_to_openai_dict(message)

    assert openai_dict == {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [
            {"id": "123", "type": "function", "function": {"name": "test_tool", "arguments": '{"foo": "bar"}'}}
        ],
    }


@pytest.mark.asyncio
async def test_convert_message_to_openai_dict_with_builtin_tool_call():
    message = CompletionMessage(
        content="",
        tool_calls=[
            ToolCall(
                call_id="123",
                tool_name=BuiltinTool.brave_search,
                arguments_json='{"foo": "bar"}',
                arguments={"foo": "bar"},
            )
        ],
        stop_reason=StopReason.end_of_turn,
    )

    openai_dict = await convert_message_to_openai_dict(message)

    assert openai_dict == {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "tool_calls": [
            {"id": "123", "type": "function", "function": {"name": "brave_search", "arguments": '{"foo": "bar"}'}}
        ],
    }


@pytest.mark.asyncio
async def test_openai_messages_to_messages_with_content_str():
    openai_messages = [
        OpenAISystemMessageParam(content="system message"),
        OpenAIUserMessageParam(content="user message"),
        OpenAIAssistantMessageParam(content="assistant message"),
    ]

    llama_messages = openai_messages_to_messages(openai_messages)
    assert len(llama_messages) == 3
    assert isinstance(llama_messages[0], SystemMessage)
    assert isinstance(llama_messages[1], UserMessage)
    assert isinstance(llama_messages[2], CompletionMessage)
    assert llama_messages[0].content == "system message"
    assert llama_messages[1].content == "user message"
    assert llama_messages[2].content == "assistant message"


@pytest.mark.asyncio
async def test_openai_messages_to_messages_with_content_list():
    openai_messages = [
        OpenAISystemMessageParam(content=[OpenAIChatCompletionContentPartTextParam(text="system message")]),
        OpenAIUserMessageParam(content=[OpenAIChatCompletionContentPartTextParam(text="user message")]),
        OpenAIAssistantMessageParam(content=[OpenAIChatCompletionContentPartTextParam(text="assistant message")]),
    ]

    llama_messages = openai_messages_to_messages(openai_messages)
    assert len(llama_messages) == 3
    assert isinstance(llama_messages[0], SystemMessage)
    assert isinstance(llama_messages[1], UserMessage)
    assert isinstance(llama_messages[2], CompletionMessage)
    assert llama_messages[0].content[0].text == "system message"
    assert llama_messages[1].content[0].text == "user message"
    assert llama_messages[2].content[0].text == "assistant message"
