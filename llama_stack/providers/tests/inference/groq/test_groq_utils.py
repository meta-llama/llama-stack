# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from groq.types.chat.chat_completion import ChatCompletion, Choice
from groq.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as StreamChoice,
    ChoiceDelta,
)
from groq.types.chat.chat_completion_message import ChatCompletionMessage

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponseEventType,
    CompletionMessage,
    StopReason,
    SystemMessage,
    UserMessage,
)
from llama_stack.providers.remote.inference.groq.groq_utils import (
    convert_chat_completion_request,
    convert_chat_completion_response,
    convert_chat_completion_response_stream,
)


class TestConvertChatCompletionRequest:
    def test_sets_model(self):
        request = self._dummy_chat_completion_request()
        request.model = "Llama-3.2-3B"

        converted = convert_chat_completion_request(request)

        assert converted["model"] == "Llama-3.2-3B"

    def test_converts_user_message(self):
        request = self._dummy_chat_completion_request()
        request.messages = [UserMessage(content="Hello World")]

        converted = convert_chat_completion_request(request)

        assert converted["messages"] == [
            {"role": "user", "content": "Hello World"},
        ]

    def test_converts_system_message(self):
        request = self._dummy_chat_completion_request()
        request.messages = [SystemMessage(content="You are a helpful assistant.")]

        converted = convert_chat_completion_request(request)

        assert converted["messages"] == [
            {"role": "system", "content": "You are a helpful assistant."},
        ]

    def test_converts_completion_message(self):
        request = self._dummy_chat_completion_request()
        request.messages = [
            UserMessage(content="Hello World"),
            CompletionMessage(
                content="Hello World! How can I help you today?",
                stop_reason=StopReason.end_of_message,
            ),
        ]

        converted = convert_chat_completion_request(request)

        assert converted["messages"] == [
            {"role": "user", "content": "Hello World"},
            {"role": "assistant", "content": "Hello World! How can I help you today?"},
        ]

    def test_does_not_include_logprobs(self):
        request = self._dummy_chat_completion_request()
        request.logprobs = True

        with pytest.warns(Warning) as warnings:
            converted = convert_chat_completion_request(request)

        assert "logprobs are not supported yet" in warnings[0].message.args[0]
        assert converted.get("logprobs") is None

    def test_does_not_include_response_format(self):
        request = self._dummy_chat_completion_request()
        request.response_format = {
            "type": "json_object",
            "json_schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                },
            },
        }

        with pytest.warns(Warning) as warnings:
            converted = convert_chat_completion_request(request)

        assert "response_format is not supported yet" in warnings[0].message.args[0]
        assert converted.get("response_format") is None

    def test_does_not_include_repetition_penalty(self):
        request = self._dummy_chat_completion_request()
        request.sampling_params.repetition_penalty = 1.5

        with pytest.warns(Warning) as warnings:
            converted = convert_chat_completion_request(request)

        assert "repetition_penalty is not supported" in warnings[0].message.args[0]
        assert converted.get("repetition_penalty") is None
        assert converted.get("frequency_penalty") is None

    def test_includes_stream(self):
        request = self._dummy_chat_completion_request()
        request.stream = True

        converted = convert_chat_completion_request(request)

        assert converted["stream"] is True

    def test_if_max_tokens_is_0_then_it_is_not_included(self):
        request = self._dummy_chat_completion_request()
        # 0 is the default value for max_tokens
        # So we assume that if it's 0, the user didn't set it
        request.sampling_params.max_tokens = 0

        converted = convert_chat_completion_request(request)

        assert converted.get("max_tokens") is None

    def test_includes_max_tokens_if_set(self):
        request = self._dummy_chat_completion_request()
        request.sampling_params.max_tokens = 100

        converted = convert_chat_completion_request(request)

        assert converted["max_tokens"] == 100

    def _dummy_chat_completion_request(self):
        return ChatCompletionRequest(
            model="Llama-3.2-3B",
            messages=[UserMessage(content="Hello World")],
        )

    def test_includes_temperature(self):
        request = self._dummy_chat_completion_request()
        request.sampling_params.temperature = 0.5

        converted = convert_chat_completion_request(request)

        assert converted["temperature"] == 0.5

    def test_includes_top_p(self):
        request = self._dummy_chat_completion_request()
        request.sampling_params.top_p = 0.95

        converted = convert_chat_completion_request(request)

        assert converted["top_p"] == 0.95


class TestConvertNonStreamChatCompletionResponse:
    def test_returns_response(self):
        response = self._dummy_chat_completion_response()
        response.choices[0].message.content = "Hello World"

        converted = convert_chat_completion_response(response)

        assert converted.completion_message.content == "Hello World"

    def test_maps_stop_to_end_of_message(self):
        response = self._dummy_chat_completion_response()
        response.choices[0].finish_reason = "stop"

        converted = convert_chat_completion_response(response)

        assert converted.completion_message.stop_reason == StopReason.end_of_turn

    def test_maps_length_to_end_of_message(self):
        response = self._dummy_chat_completion_response()
        response.choices[0].finish_reason = "length"

        converted = convert_chat_completion_response(response)

        assert converted.completion_message.stop_reason == StopReason.out_of_tokens

    def _dummy_chat_completion_response(self):
        return ChatCompletion(
            id="chatcmpl-123",
            model="Llama-3.2-3B",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="Hello World"
                    ),
                    finish_reason="stop",
                )
            ],
            created=1729382400,
            object="chat.completion",
        )


class TestConvertStreamChatCompletionResponse:
    @pytest.mark.asyncio
    async def test_returns_stream(self):
        def chat_completion_stream():
            messages = ["Hello ", "World ", " !"]
            for i, message in enumerate(messages):
                chunk = self._dummy_chat_completion_chunk()
                chunk.choices[0].delta.content = message
                if i == len(messages) - 1:
                    chunk.choices[0].finish_reason = "stop"
                else:
                    chunk.choices[0].finish_reason = None
                yield chunk

            chunk = self._dummy_chat_completion_chunk()
            chunk.choices[0].delta.content = None
            chunk.choices[0].finish_reason = "stop"
            yield chunk

        stream = chat_completion_stream()
        converted = convert_chat_completion_response_stream(stream)

        iter = converted.__aiter__()
        chunk = await iter.__anext__()
        assert chunk.event.event_type == ChatCompletionResponseEventType.start
        assert chunk.event.delta == "Hello "

        chunk = await iter.__anext__()
        assert chunk.event.event_type == ChatCompletionResponseEventType.progress
        assert chunk.event.delta == "World "

        chunk = await iter.__anext__()
        assert chunk.event.event_type == ChatCompletionResponseEventType.progress
        assert chunk.event.delta == " !"

        # Dummy chunk to ensure the last chunk is really the end of the stream
        # This one technically maps to Groq's final "stop" chunk
        chunk = await iter.__anext__()
        assert chunk.event.event_type == ChatCompletionResponseEventType.progress
        assert chunk.event.delta == ""

        chunk = await iter.__anext__()
        assert chunk.event.event_type == ChatCompletionResponseEventType.complete
        assert chunk.event.delta == ""
        assert chunk.event.stop_reason == StopReason.end_of_turn

        with pytest.raises(StopAsyncIteration):
            await iter.__anext__()

    def _dummy_chat_completion_chunk(self):
        return ChatCompletionChunk(
            id="chatcmpl-123",
            model="Llama-3.2-3B",
            choices=[
                StreamChoice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hello World"),
                )
            ],
            created=1729382400,
            object="chat.completion.chunk",
            x_groq=None,
        )
