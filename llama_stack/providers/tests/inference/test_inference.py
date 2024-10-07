# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import itertools
import os
from datetime import datetime

import pytest
import pytest_asyncio
import yaml

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.resolver import resolve_impls_with_routing


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


async def stack_impls(model):
    if "PROVIDER_CONFIG" not in os.environ:
        raise ValueError(
            "You must set PROVIDER_CONFIG to a YAML file containing provider config"
        )

    with open(os.environ["PROVIDER_CONFIG"], "r") as f:
        config_dict = yaml.safe_load(f)

    if "providers" not in config_dict:
        raise ValueError("Config file should contain a `providers` key")

    providers_by_id = {x["provider_id"]: x for x in config_dict["providers"]}
    if len(providers_by_id) == 0:
        raise ValueError("No providers found in config file")

    if "PROVIDER_ID" in os.environ:
        provider_id = os.environ["PROVIDER_ID"]
        if provider_id not in providers_by_id:
            raise ValueError(f"Provider ID {provider_id} not found in config file")
        provider = providers_by_id[provider_id]
    else:
        provider = list(providers_by_id.values())[0]
        print(f"No provider ID specified, picking first {provider['provider_id']}")

    config_dict = dict(
        built_at=datetime.now(),
        image_name="test-fixture",
        apis=[
            Api.inference,
            Api.models,
        ],
        providers=dict(
            inference=[
                Provider(**provider),
            ]
        ),
        models=[
            ModelDef(
                identifier=model,
                llama_model=model,
                provider_id=provider["provider_id"],
            )
        ],
        shields=[],
        memory_banks=[],
    )
    run_config = parse_and_maybe_upgrade_config(config_dict)
    impls = await resolve_impls_with_routing(run_config)
    return impls


# This is going to create multiple Stack impls without tearing down the previous one
# Fix that!
@pytest_asyncio.fixture(
    scope="session",
    params=[
        {"model": Llama_8B},
        {"model": Llama_3B},
    ],
)
async def inference_settings(request):
    model = request.param["model"]
    impls = await stack_impls(model)
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

    stop_reason = get_expected_stop_reason(inference_settings["common_params"]["model"])
    assert message.stop_reason == stop_reason
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

    end = grouped[ChatCompletionResponseEventType.complete][0]
    expected_stop_reason = get_expected_stop_reason(
        inference_settings["common_params"]["model"]
    )
    assert end.event.stop_reason == expected_stop_reason

    model = inference_settings["common_params"]["model"]
    if "Llama3.1" in model:
        assert all(
            isinstance(chunk.event.delta, ToolCallDelta)
            for chunk in grouped[ChatCompletionResponseEventType.progress]
        )
        first = grouped[ChatCompletionResponseEventType.progress][0]
        assert first.event.delta.parse_status == ToolCallParseStatus.started

    last = grouped[ChatCompletionResponseEventType.progress][-1]
    assert last.event.stop_reason == expected_stop_reason
    assert last.event.delta.parse_status == ToolCallParseStatus.success
    assert isinstance(last.event.delta.content, ToolCall)

    call = last.event.delta.content
    assert call.tool_name == "get_weather"
    assert "location" in call.arguments
    assert "San Francisco" in call.arguments["location"]
