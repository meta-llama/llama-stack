# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import itertools

import pytest
import pytest_asyncio

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.providers.tests.resolver import resolve_impls_for_test


def group_chunks(response):
    return {
        event_type: list(group)
        for event_type, group in itertools.groupby(
            response, key=lambda chunk: chunk.event.event_type
        )
    }


Llama_8B = "Llama3.1-8B-Instruct"
Llama_3B = "Llama3.2-3B-Instruct"


def get_expected_stop_reason(model: str):
    return StopReason.end_of_message if "Llama3.1" in model else StopReason.end_of_turn


# This is going to create multiple Stack impls without tearing down the previous one
# Fix that!
@pytest_asyncio.fixture(
    scope="session",
    params=[
        {"model": Llama_8B},
        {"model": Llama_3B},
    ],
    ids=lambda d: d["model"],
)
async def inference_settings(request):
    model = request.param["model"]
    impls = await resolve_impls_for_test(
        Api.inference,
        models=[
            ModelDef(
                identifier=model,
                llama_model=model,
                provider_id="",
            )
        ],
    )

    return {
        "impl": impls[Api.inference],
        "common_params": {
            "model": model,
            "tool_choice": ToolChoice.auto,
            "tool_prompt_format": (
                ToolPromptFormat.json
                if "Llama3.1" in model
                else ToolPromptFormat.python_list
            ),
        },
    }


@pytest.fixture
def sample_messages():
    return [
        SystemMessage(content="You are a helpful assistant."),
        UserMessage(content="What's the weather like today?"),
    ]


@pytest.fixture
def sample_tool_definition():
    return ToolDefinition(
        tool_name="get_weather",
        description="Get the current weather",
        parameters={
            "location": ToolParamDefinition(
                param_type="string",
                description="The city and state, e.g. San Francisco, CA",
            ),
        },
    )


@pytest.mark.asyncio
async def test_chat_completion_non_streaming(inference_settings, sample_messages):
    inference_impl = inference_settings["impl"]
    response = [
        r
        async for r in inference_impl.chat_completion(
            messages=sample_messages,
            stream=False,
            **inference_settings["common_params"],
        )
    ]

    assert len(response) == 1
    assert isinstance(response[0], ChatCompletionResponse)
    assert response[0].completion_message.role == "assistant"
    assert isinstance(response[0].completion_message.content, str)
    assert len(response[0].completion_message.content) > 0


@pytest.mark.asyncio
async def test_chat_completion_streaming(inference_settings, sample_messages):
    inference_impl = inference_settings["impl"]
    response = [
        r
        async for r in inference_impl.chat_completion(
            messages=sample_messages,
            stream=True,
            **inference_settings["common_params"],
        )
    ]

    assert len(response) > 0
    assert all(
        isinstance(chunk, ChatCompletionResponseStreamChunk) for chunk in response
    )
    grouped = group_chunks(response)
    assert len(grouped[ChatCompletionResponseEventType.start]) == 1
    assert len(grouped[ChatCompletionResponseEventType.progress]) > 0
    assert len(grouped[ChatCompletionResponseEventType.complete]) == 1

    end = grouped[ChatCompletionResponseEventType.complete][0]
    assert end.event.stop_reason == StopReason.end_of_turn


@pytest.mark.asyncio
async def test_chat_completion_with_tool_calling(
    inference_settings,
    sample_messages,
    sample_tool_definition,
):
    inference_impl = inference_settings["impl"]
    messages = sample_messages + [
        UserMessage(
            content="What's the weather like in San Francisco?",
        )
    ]

    response = [
        r
        async for r in inference_impl.chat_completion(
            messages=messages,
            tools=[sample_tool_definition],
            stream=False,
            **inference_settings["common_params"],
        )
    ]

    assert len(response) == 1
    assert isinstance(response[0], ChatCompletionResponse)

    message = response[0].completion_message

    # This is not supported in most providers :/ they don't return eom_id / eot_id
    # stop_reason = get_expected_stop_reason(inference_settings["common_params"]["model"])
    # assert message.stop_reason == stop_reason
    assert message.tool_calls is not None
    assert len(message.tool_calls) > 0

    call = message.tool_calls[0]
    assert call.tool_name == "get_weather"
    assert "location" in call.arguments
    assert "San Francisco" in call.arguments["location"]


@pytest.mark.asyncio
async def test_chat_completion_with_tool_calling_streaming(
    inference_settings,
    sample_messages,
    sample_tool_definition,
):
    inference_impl = inference_settings["impl"]
    messages = sample_messages + [
        UserMessage(
            content="What's the weather like in San Francisco?",
        )
    ]

    response = [
        r
        async for r in inference_impl.chat_completion(
            messages=messages,
            tools=[sample_tool_definition],
            stream=True,
            **inference_settings["common_params"],
        )
    ]

    assert len(response) > 0
    assert all(
        isinstance(chunk, ChatCompletionResponseStreamChunk) for chunk in response
    )
    grouped = group_chunks(response)
    assert len(grouped[ChatCompletionResponseEventType.start]) == 1
    assert len(grouped[ChatCompletionResponseEventType.progress]) > 0
    assert len(grouped[ChatCompletionResponseEventType.complete]) == 1

    # This is not supported in most providers :/ they don't return eom_id / eot_id
    # expected_stop_reason = get_expected_stop_reason(
    #     inference_settings["common_params"]["model"]
    # )
    # end = grouped[ChatCompletionResponseEventType.complete][0]
    # assert end.event.stop_reason == expected_stop_reason

    model = inference_settings["common_params"]["model"]
    if "Llama3.1" in model:
        assert all(
            isinstance(chunk.event.delta, ToolCallDelta)
            for chunk in grouped[ChatCompletionResponseEventType.progress]
        )
        first = grouped[ChatCompletionResponseEventType.progress][0]
        assert first.event.delta.parse_status == ToolCallParseStatus.started

    last = grouped[ChatCompletionResponseEventType.progress][-1]
    # assert last.event.stop_reason == expected_stop_reason
    assert last.event.delta.parse_status == ToolCallParseStatus.success
    assert isinstance(last.event.delta.content, ToolCall)

    call = last.event.delta.content
    assert call.tool_name == "get_weather"
    assert "location" in call.arguments
    assert "San Francisco" in call.arguments["location"]
