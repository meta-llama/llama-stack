# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_models.llama3.api.datatypes import TokenLogProbs, ToolCall

from llama_stack.apis.inference import Inference
from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.asyncio


async def test_content(
    mock_health: HTTPXMock,
    httpx_mock: HTTPXMock,
    client: Inference,
    base_url: str,
) -> None:
    """
    Test that response content makes it through to the completion message.
    """
    httpx_mock.add_response(
        url=f"{base_url}/v1/chat/completions",
        json={
            "id": "mock-id",
            "created": 1234567890,
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "RESPONSE"},
                    "finish_reason": "length",
                }
            ],
        },
        status_code=200,
    )

    client = await client

    response = await client.chat_completion(
        model="Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "BOGUS"}],
        stream=False,
    )
    assert response.completion_message.content == "RESPONSE"


async def test_logprobs(
    mock_health: HTTPXMock,
    httpx_mock: HTTPXMock,
    client: Inference,
    base_url: str,
) -> None:
    """
    Test that logprobs are parsed correctly.
    """
    httpx_mock.add_response(
        url=f"{base_url}/v1/chat/completions",
        json={
            "id": "mock-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello there"},
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hello",
                                "logprob": -0.1,
                                "bytes": [72, 101, 108, 108, 111],
                                "top_logprobs": [
                                    {"token": "Hello", "logprob": -0.1},
                                    {"token": "Hi", "logprob": -1.2},
                                    {"token": "Greetings", "logprob": -2.1},
                                ],
                            },
                            {
                                "token": "there",
                                "logprob": -0.2,
                                "bytes": [116, 104, 101, 114, 101],
                                "top_logprobs": [
                                    {"token": "there", "logprob": -0.2},
                                    {"token": "here", "logprob": -1.3},
                                    {"token": "where", "logprob": -2.2},
                                ],
                            },
                        ]
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        status_code=200,
    )

    client = await client

    response = await client.chat_completion(
        model="Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        logprobs={"top_k": 3},
        stream=False,
    )

    assert response.logprobs == [
        TokenLogProbs(
            logprobs_by_token={
                "Hello": -0.1,
                "Hi": -1.2,
                "Greetings": -2.1,
            }
        ),
        TokenLogProbs(
            logprobs_by_token={
                "there": -0.2,
                "here": -1.3,
                "where": -2.2,
            }
        ),
    ]


async def test_tools(
    mock_health: HTTPXMock,
    httpx_mock: HTTPXMock,
    client: Inference,
    base_url: str,
) -> None:
    """
    Test that tools are passed correctly.
    """
    httpx_mock.add_response(
        url=f"{base_url}/v1/chat/completions",
        json={
            "id": "mock-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool-id",
                                "type": "function",
                                "function": {
                                    "name": "magic",
                                    "arguments": {"input": 3},
                                },
                            },
                            {
                                "id": "tool-id!",
                                "type": "function",
                                "function": {
                                    "name": "magic!",
                                    "arguments": {"input": 42},
                                },
                            },
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
        },
        status_code=200,
    )

    client = await client

    response = await client.chat_completion(
        model="Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "Hello"}],
        stream=False,
    )

    assert response.completion_message.tool_calls == [
        ToolCall(
            call_id="tool-id",
            tool_name="magic",
            arguments={"input": 3},
        ),
        ToolCall(
            call_id="tool-id!",
            tool_name="magic!",
            arguments={"input": 42},
        ),
    ]


# TODO(mf): test stream=True for each case
