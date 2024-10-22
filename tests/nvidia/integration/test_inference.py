# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import itertools
from typing import Generator, List, Tuple

import pytest

from llama_stack.apis.inference import (
    ChatCompletionResponse,
    CompletionMessage,
    Inference,
    Message,
    StopReason,
    SystemMessage,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.providers.adapters.inference.nvidia import (
    get_adapter_impl,
    NVIDIAConfig,
)

pytestmark = pytest.mark.asyncio


# TODO(mf): test bad creds raises PermissionError
# TODO(mf): test bad params, e.g. max_tokens=0 raises ValidationError
# TODO(mf): test bad model name raises ValueError
# TODO(mf): test short timeout raises TimeoutError
# TODO(mf): new file, test cli model listing
# TODO(mf): test streaming
# TODO(mf): test tool calls w/ tool_choice


def message_combinations(
    length: int,
) -> Generator[Tuple[List[Message], str], None, None]:
    """
    Generate all possible combinations of message types of given length.
    """
    message_types = [
        UserMessage,
        SystemMessage,
        ToolResponseMessage,
        CompletionMessage,
    ]
    for count in range(1, length + 1):
        for combo in itertools.product(message_types, repeat=count):
            messages = []
            for i, msg in enumerate(combo):
                if msg == ToolResponseMessage:
                    messages.append(
                        msg(
                            content=f"Message {i + 1}",
                            call_id=f"call_{i + 1}",
                            tool_name=f"tool_{i + 1}",
                        )
                    )
                elif msg == CompletionMessage:
                    messages.append(
                        msg(content=f"Message {i + 1}", stop_reason="end_of_message")
                    )
                else:
                    messages.append(msg(content=f"Message {i + 1}"))
            id_str = "-".join([msg.__name__ for msg in combo])
            yield messages, id_str


@pytest.mark.parametrize("combo", message_combinations(3), ids=lambda x: x[1])
async def test_chat_completion_messages(
    client: Inference,
    model: str,
    combo: Tuple[List[Message], str],
):
    """
    Test the chat completion endpoint with different message combinations.
    """
    client = await client
    messages, _ = combo

    response = await client.chat_completion(
        model=model,
        messages=messages,
        stream=False,
    )

    assert isinstance(response, ChatCompletionResponse)
    assert isinstance(response.completion_message.content, str)
    # we're not testing accuracy, so no assertions on the result.completion_message.content
    assert response.completion_message.role == "assistant"
    assert isinstance(response.completion_message.stop_reason, StopReason)
    assert response.completion_message.tool_calls == []


async def test_bad_base_url(
    model: str,
):
    """
    Test that a bad base_url raises a ConnectionError.
    """
    client = await get_adapter_impl(
        NVIDIAConfig(
            base_url="http://localhost:32123",
        ),
        {},
    )

    with pytest.raises(ConnectionError):
        await client.chat_completion(
            model=model,
            messages=[UserMessage(content="Hello")],
            stream=False,
        )
