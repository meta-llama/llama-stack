# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Type

from unittest.mock import create_autospec, MagicMock, Mock, patch

import pytest
from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
    Inference,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.models import Model
from llama_stack.distribution.routers import ModelsRoutingTable
from llama_stack.distribution.routers.routers import InferenceRouter
from llama_stack.providers.remote.inference.ollama.ollama import OllamaInferenceAdapter


class Stubs:
    completion_stub_matchers = {
        "stream=False": {
            "content=Micheael Jordan is born in ": CompletionResponse(
                content="1963",
                stop_reason="end_of_message",
                logprobs=None,
            )
        },
        "stream=True": {
            "content=Roses are red,": CompletionResponseStreamChunk(
                delta="", stop_reason="out_of_tokens", logprobs=None
            )
        },
    }

    @staticmethod
    async def process_completion(*args, **kwargs):
        if kwargs["stream"]:
            stream_mock = MagicMock()
            stream_mock.__aiter__.return_value = [
                Stubs.completion_stub_matchers["stream=True"][
                    f"content={kwargs['content']}"
                ]
            ]
            return stream_mock
        return Stubs.completion_stub_matchers["stream=False"][
            f"content={kwargs['content']}"
        ]

    chat_completion_stub_matchers = {
        "stream=False": {
            "content=You are a helpful assistant.|What's the weather like today?": ChatCompletionResponse(
                completion_message=CompletionMessage(
                    role="assistant",
                    content="Hello world",
                    stop_reason="end_of_message",
                )
            ),
            "content=You are a helpful assistant.|What's the weather like today?|What's the weather like in San Francisco?": ChatCompletionResponse(
                completion_message=CompletionMessage(
                    role="assistant",
                    content="Hello world",
                    stop_reason="end_of_message",
                    tool_calls=[
                        ToolCall(
                            call_id="get_weather",
                            tool_name="get_weather",
                            arguments={"location": "San Francisco"},
                        )
                    ],
                )
            ),
        },
        "stream=True": {
            "content=You are a helpful assistant.|What's the weather like today?": [
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.start,
                        delta="Hello",
                    )
                ),
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta="world",
                    )
                ),
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.complete,
                        delta="this is a test",
                        stop_reason="end_of_turn",
                    )
                ),
            ],
            "content=You are a helpful assistant.|What's the weather like today?|What's the weather like in San Francisco?": [
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.start,
                        delta=ToolCallDelta(
                            content=ToolCall(
                                call_id="get_weather",
                                tool_name="get_weather",
                                arguments={"location": "San Francisco"},
                            ),
                            parse_status=ToolCallParseStatus.success,
                        ),
                    ),
                ),
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            content=ToolCall(
                                call_id="get_weather",
                                tool_name="get_weather",
                                arguments={"location": "San Francisco"},
                            ),
                            parse_status=ToolCallParseStatus.success,
                        ),
                    ),
                ),
                ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.complete,
                        delta=ToolCallDelta(
                            content=ToolCall(
                                call_id="get_weather",
                                tool_name="get_weather",
                                arguments={"location": "San Francisco"},
                            ),
                            parse_status=ToolCallParseStatus.success,
                        ),
                    )
                ),
            ],
        },
    }

    @staticmethod
    async def chat_completion(*args, **kwargs):
        query_content = "|".join([msg.content for msg in kwargs["messages"]])
        if kwargs["stream"]:
            stream_mock = MagicMock()
            stream_mock.__aiter__.return_value = Stubs.chat_completion_stub_matchers[
                "stream=True"
            ][f"content={query_content}"]
            return stream_mock
        return Stubs.chat_completion_stub_matchers["stream=False"][
            f"content={query_content}"
        ]


def setup_models_stubs(model_mock: Model, routing_table_mock: Type[ModelsRoutingTable]):
    routing_table_mock.return_value.list_models.return_value = [model_mock]


def setup_provider_stubs(
    model_mock: Model, routing_table_mock: Type[ModelsRoutingTable]
):
    provider_mock = Mock()
    provider_mock.__provider_spec__ = Mock()
    provider_mock.__provider_spec__.provider_type = model_mock.provider_type
    routing_table_mock.return_value.get_provider_impl.return_value = provider_mock


def setup_inference_router_stubs(adapter_class: Type[Inference]):
    # Set up competion stubs
    InferenceRouter.completion = create_autospec(adapter_class.completion)
    InferenceRouter.completion.side_effect = Stubs.process_completion

    # Set up chat completion stubs
    InferenceRouter.chat_completion = create_autospec(adapter_class.chat_completion)
    InferenceRouter.chat_completion.side_effect = Stubs.chat_completion


@pytest.fixture(scope="session")
def inference_ollama_mocks(inference_model):
    with patch(
        "llama_stack.providers.remote.inference.ollama.get_adapter_impl",
        autospec=True,
    ) as get_adapter_impl_mock, patch(
        "llama_stack.distribution.routers.ModelsRoutingTable",
        autospec=True,
    ) as ModelsRoutingTableMock:  # noqa N806
        model_mock = create_autospec(Model)
        model_mock.identifier = inference_model
        model_mock.provider_id = "ollama"
        model_mock.provider_type = "remote::ollama"

        setup_models_stubs(model_mock, ModelsRoutingTableMock)
        setup_provider_stubs(model_mock, ModelsRoutingTableMock)
        setup_inference_router_stubs(OllamaInferenceAdapter)

        impl_mock = create_autospec(OllamaInferenceAdapter)
        get_adapter_impl_mock.return_value = impl_mock
        yield
