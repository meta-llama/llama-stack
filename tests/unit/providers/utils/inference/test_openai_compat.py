# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.apis.common.content_types import TextContentItem
from llama_stack.apis.inference.inference import CompletionMessage, UserMessage
from llama_stack.models.llama.datatypes import StopReason, ToolCall
from llama_stack.providers.utils.inference.openai_compat import convert_message_to_openai_dict


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
