# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List

import pytest
from llama_models.llama3.api.datatypes import StopReason

from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
)
from llama_stack.providers.adapters.inference.nvidia._openai_utils import (
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
)
from openai.types.chat import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionTokenLogprob,
)
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.chat.chat_completion_chunk import (
    Choice as ChoiceChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_token_logprob import TopLogprob


def test_convert_openai_chat_completion_choice_basic():
    response = Choice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content="Hello, world!",
        ),
        finish_reason="stop",
    )
    result = convert_openai_chat_completion_choice(response)
    assert isinstance(result, ChatCompletionResponse)
    assert result.completion_message.content == "Hello, world!"
    assert result.completion_message.stop_reason == StopReason.end_of_turn
    assert result.completion_message.tool_calls == []
    assert result.logprobs is None


def test_convert_openai_chat_completion_choice_basic_with_tool_calls():
    response = Choice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content="Hello, world!",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="tool_call_id",
                    type="function",
                    function={
                        "name": "test_function",
                        "arguments": '{"test_args": "test_value"}',
                    },
                )
            ],
        ),
        finish_reason="tool_calls",
    )

    result = convert_openai_chat_completion_choice(response)
    assert isinstance(result, ChatCompletionResponse)
    assert result.completion_message.content == "Hello, world!"
    assert result.completion_message.stop_reason == StopReason.end_of_message
    assert len(result.completion_message.tool_calls) == 1
    assert result.completion_message.tool_calls[0].tool_name == "test_function"
    assert result.completion_message.tool_calls[0].arguments == {
        "test_args": "test_value"
    }
    assert result.logprobs is None


def test_convert_openai_chat_completion_choice_basic_with_logprobs():
    response = Choice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content="Hello world",
        ),
        finish_reason="stop",
        logprobs=ChoiceLogprobs(
            content=[
                ChatCompletionTokenLogprob(
                    token="Hello",
                    logprob=-1.0,
                    bytes=[72, 101, 108, 108, 111],
                    top_logprobs=[
                        TopLogprob(
                            token="Hello", logprob=-1.0, bytes=[72, 101, 108, 108, 111]
                        ),
                        TopLogprob(
                            token="Greetings",
                            logprob=-1.5,
                            bytes=[71, 114, 101, 101, 116, 105, 110, 103, 115],
                        ),
                    ],
                ),
                ChatCompletionTokenLogprob(
                    token="world",
                    logprob=-1.5,
                    bytes=[119, 111, 114, 108, 100],
                    top_logprobs=[
                        TopLogprob(
                            token="world", logprob=-1.5, bytes=[119, 111, 114, 108, 100]
                        ),
                        TopLogprob(
                            token="planet",
                            logprob=-2.0,
                            bytes=[112, 108, 97, 110, 101, 116],
                        ),
                    ],
                ),
            ]
        ),
    )

    result = convert_openai_chat_completion_choice(response)
    assert isinstance(result, ChatCompletionResponse)
    assert result.completion_message.content == "Hello world"
    assert result.completion_message.stop_reason == StopReason.end_of_turn
    assert result.completion_message.tool_calls == []
    assert result.logprobs is not None
    assert len(result.logprobs) == 2
    assert len(result.logprobs[0].logprobs_by_token) == 2
    assert result.logprobs[0].logprobs_by_token["Hello"] == -1.0
    assert result.logprobs[0].logprobs_by_token["Greetings"] == -1.5
    assert len(result.logprobs[1].logprobs_by_token) == 2
    assert result.logprobs[1].logprobs_by_token["world"] == -1.5
    assert result.logprobs[1].logprobs_by_token["planet"] == -2.0


def test_convert_openai_chat_completion_choice_missing_message():
    response = Choice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content="Hello, world!",
        ),
        finish_reason="stop",
    )

    response.message = None
    with pytest.raises(
        AssertionError, match="error in server response: message not found"
    ):
        convert_openai_chat_completion_choice(response)

    del response.message
    with pytest.raises(
        AssertionError, match="error in server response: message not found"
    ):
        convert_openai_chat_completion_choice(response)


def test_convert_openai_chat_completion_choice_missing_finish_reason():
    response = Choice(
        index=0,
        message=ChatCompletionMessage(
            role="assistant",
            content="Hello, world!",
        ),
        finish_reason="stop",
    )

    response.finish_reason = None
    with pytest.raises(
        AssertionError, match="error in server response: finish_reason not found"
    ):
        convert_openai_chat_completion_choice(response)

    del response.finish_reason
    with pytest.raises(
        AssertionError, match="error in server response: finish_reason not found"
    ):
        convert_openai_chat_completion_choice(response)


# we want to test convert_openai_chat_completion_stream
# we need to produce a stream of OpenAIChatCompletionChunk
# streams to produce -
#  0. basic stream with one chunk, should produce 3 (start, progress, complete)
#  1. stream with 3 chunks, should produce 5 events (start, progress, progress, progress, complete)
#  2. stream with a tool call, should produce 4 events (start, progress w/ tool_call, complete)


@pytest.mark.asyncio
async def test_convert_openai_chat_completion_stream_basic():
    chunks = [
        OpenAIChatCompletionChunk(
            id="1",
            created=1234567890,
            model="mock-model",
            object="chat.completion.chunk",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="Hello, world!",
                    ),
                    finish_reason="stop",
                )
            ],
        )
    ]

    async def async_generator_from_list(items: List) -> AsyncGenerator:
        for item in items:
            yield item

    results = [
        result
        async for result in convert_openai_chat_completion_stream(
            async_generator_from_list(chunks)
        )
    ]

    assert len(results) == 2
    assert all(
        isinstance(result, ChatCompletionResponseStreamChunk) for result in results
    )
    assert results[0].event.event_type == ChatCompletionResponseEventType.start
    assert results[0].event.delta == "Hello, world!"
    assert results[1].event.event_type == ChatCompletionResponseEventType.complete
    assert results[1].event.stop_reason == StopReason.end_of_turn


@pytest.mark.asyncio
async def test_convert_openai_chat_completion_stream_basic_empty():
    chunks = [
        OpenAIChatCompletionChunk(
            id="1",
            created=1234567890,
            model="mock-model",
            object="chat.completion.chunk",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                    ),
                    finish_reason="stop",
                )
            ],
        ),
        OpenAIChatCompletionChunk(
            id="1",
            created=1234567890,
            model="mock-model",
            object="chat.completion.chunk",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="Hello, world!",
                    ),
                    finish_reason="stop",
                )
            ],
        ),
    ]

    async def async_generator_from_list(items: List) -> AsyncGenerator:
        for item in items:
            yield item

    results = [
        result
        async for result in convert_openai_chat_completion_stream(
            async_generator_from_list(chunks)
        )
    ]

    print(results)

    assert len(results) == 3
    assert all(
        isinstance(result, ChatCompletionResponseStreamChunk) for result in results
    )
    assert results[0].event.event_type == ChatCompletionResponseEventType.start
    assert results[1].event.event_type == ChatCompletionResponseEventType.progress
    assert results[1].event.delta == "Hello, world!"
    assert results[2].event.event_type == ChatCompletionResponseEventType.complete
    assert results[2].event.stop_reason == StopReason.end_of_turn


@pytest.mark.asyncio
async def test_convert_openai_chat_completion_stream_multiple_chunks():
    chunks = [
        OpenAIChatCompletionChunk(
            id="1",
            created=1234567890,
            model="mock-model",
            object="chat.completion.chunk",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="Hello, world!",
                    ),
                    # finish_reason="continue",
                )
            ],
        ),
        OpenAIChatCompletionChunk(
            id="2",
            created=1234567891,
            model="mock-model",
            object="chat.completion.chunk",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="How are you?",
                    ),
                    # finish_reason="continue",
                )
            ],
        ),
        OpenAIChatCompletionChunk(
            id="3",
            created=1234567892,
            model="mock-model",
            object="chat.completion.chunk",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="I'm good, thanks!",
                    ),
                    finish_reason="stop",
                )
            ],
        ),
    ]

    async def async_generator_from_list(items: List) -> AsyncGenerator:
        for item in items:
            yield item

    results = [
        result
        async for result in convert_openai_chat_completion_stream(
            async_generator_from_list(chunks)
        )
    ]

    assert len(results) == 4
    assert all(
        isinstance(result, ChatCompletionResponseStreamChunk) for result in results
    )
    assert results[0].event.event_type == ChatCompletionResponseEventType.start
    assert results[0].event.delta == "Hello, world!"
    assert not results[0].event.stop_reason
    assert results[1].event.event_type == ChatCompletionResponseEventType.progress
    assert results[1].event.delta == "How are you?"
    assert not results[1].event.stop_reason
    assert results[2].event.event_type == ChatCompletionResponseEventType.progress
    assert results[2].event.delta == "I'm good, thanks!"
    assert not results[2].event.stop_reason
    assert results[3].event.event_type == ChatCompletionResponseEventType.complete
    assert results[3].event.stop_reason == StopReason.end_of_turn


@pytest.mark.asyncio
async def test_convert_openai_chat_completion_stream_with_tool_call_and_content():
    chunks = [
        OpenAIChatCompletionChunk(
            id="1",
            created=1234567890,
            model="mock-model",
            object="chat.completion.chunk",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        content="Hello, world!",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tool_call_id",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="test_function",
                                    arguments='{"test_args": "test_value"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    async def async_generator_from_list(items: List) -> AsyncGenerator:
        for item in items:
            yield item

    results = [
        result
        async for result in convert_openai_chat_completion_stream(
            async_generator_from_list(chunks)
        )
    ]

    assert len(results) == 3
    assert all(
        isinstance(result, ChatCompletionResponseStreamChunk) for result in results
    )
    assert results[0].event.event_type == ChatCompletionResponseEventType.start
    assert results[0].event.delta == "Hello, world!"
    assert not results[0].event.stop_reason
    assert results[1].event.event_type == ChatCompletionResponseEventType.progress
    assert not isinstance(results[1].event.delta, str)
    assert results[1].event.delta.content.tool_name == "test_function"
    assert results[1].event.delta.content.arguments == {"test_args": "test_value"}
    assert not results[1].event.stop_reason
    assert results[2].event.event_type == ChatCompletionResponseEventType.complete
    assert results[2].event.stop_reason == StopReason.end_of_message


@pytest.mark.asyncio
async def test_convert_openai_chat_completion_stream_with_tool_call_and_no_content():
    chunks = [
        OpenAIChatCompletionChunk(
            id="1",
            created=1234567890,
            model="mock-model",
            object="chat.completion.chunk",
            choices=[
                ChoiceChunk(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="tool_call_id",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="test_function",
                                    arguments='{"test_args": "test_value"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )
    ]

    async def async_generator_from_list(items: List) -> AsyncGenerator:
        for item in items:
            yield item

    results = [
        result
        async for result in convert_openai_chat_completion_stream(
            async_generator_from_list(chunks)
        )
    ]

    assert len(results) == 2
    assert all(
        isinstance(result, ChatCompletionResponseStreamChunk) for result in results
    )
    assert results[0].event.event_type == ChatCompletionResponseEventType.start
    assert not isinstance(results[0].event.delta, str)
    assert results[0].event.delta.content.tool_name == "test_function"
    assert results[0].event.delta.content.arguments == {"test_args": "test_value"}
    assert not results[0].event.stop_reason
    assert results[1].event.event_type == ChatCompletionResponseEventType.complete
    assert results[1].event.stop_reason == StopReason.end_of_message
