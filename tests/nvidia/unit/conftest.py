# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import pytest

from llama_stack.apis.inference import Inference
from llama_stack.providers.adapters.inference.nvidia import (
    get_adapter_impl,
    NVIDIAConfig,
)
from pytest_httpx import HTTPXMock

pytestmark = pytest.mark.asyncio


@pytest.fixture
def base_url():
    return "http://endpoint.mocked"


@pytest.fixture
def client(base_url: str) -> Inference:
    return get_adapter_impl(
        NVIDIAConfig(
            base_url=base_url,
            api_key=os.environ.get("NVIDIA_API_KEY"),
        ),
        {},
    )


@pytest.fixture
def mock_health(
    httpx_mock: HTTPXMock,
    base_url: str,
) -> HTTPXMock:
    for path in [
        "/v1/health/live",
        "/v1/health/ready",
    ]:
        httpx_mock.add_response(
            url=f"{base_url}{path}",
            status_code=200,
        )
    return httpx_mock


@pytest.fixture
def mock_chat_completion(httpx_mock: HTTPXMock, base_url: str) -> HTTPXMock:
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
                    "message": {"role": "assistant", "content": "WORKED"},
                    "finish_reason": "length",
                }
            ],
        },
        status_code=200,
    )

    return httpx_mock
