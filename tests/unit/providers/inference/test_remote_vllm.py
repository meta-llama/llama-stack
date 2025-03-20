# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as OpenAIChoice,
)
from openai.types.chat.chat_completion_chunk import (
    ChoiceDelta as OpenAIChoiceDelta,
)
from openai.types.model import Model as OpenAIModel

from llama_stack.apis.inference import ToolChoice, ToolConfig
from llama_stack.apis.models import Model
from llama_stack.models.llama.datatypes import StopReason
from llama_stack.providers.remote.inference.vllm.config import VLLMInferenceAdapterConfig
from llama_stack.providers.remote.inference.vllm.vllm import (
    VLLMInferenceAdapter,
    _process_vllm_chat_completion_stream_response,
)

# These are unit test for the remote vllm provider
# implementation. This should only contain tests which are specific to
# the implementation details of those classes. More general
# (API-level) tests should be placed in tests/integration/inference/
#
# How to run this test:
#
# pytest tests/unit/providers/inference/test_remote_vllm.py \
# -v -s --tb=short --disable-warnings


class MockInferenceAdapterWithSleep:
    def __init__(self, sleep_time: int, response: Dict[str, Any]):
        self.httpd = None

        class DelayedRequestHandler(BaseHTTPRequestHandler):
            # ruff: noqa: N802
            def do_POST(self):
                time.sleep(sleep_time)
                self.send_response(code=200)
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

        self.request_handler = DelayedRequestHandler

    def __enter__(self):
        httpd = HTTPServer(("", 0), self.request_handler)
        self.httpd = httpd
        host, port = httpd.server_address
        httpd_thread = threading.Thread(target=httpd.serve_forever)
        httpd_thread.daemon = True  # stop server if this thread terminates
        httpd_thread.start()

        config = VLLMInferenceAdapterConfig(url=f"http://{host}:{port}")
        inference_adapter = VLLMInferenceAdapter(config)
        return inference_adapter

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


@pytest.fixture(scope="module")
def mock_openai_models_list():
    with patch("openai.resources.models.AsyncModels.list", new_callable=AsyncMock) as mock_list:
        yield mock_list


@pytest_asyncio.fixture(scope="module")
async def vllm_inference_adapter():
    config = VLLMInferenceAdapterConfig(url="http://mocked.localhost:12345")
    inference_adapter = VLLMInferenceAdapter(config)
    inference_adapter.model_store = AsyncMock()
    await inference_adapter.initialize()
    return inference_adapter


@pytest.mark.asyncio
async def test_register_model_checks_vllm(mock_openai_models_list, vllm_inference_adapter):
    async def mock_openai_models():
        yield OpenAIModel(id="foo", created=1, object="model", owned_by="test")

    mock_openai_models_list.return_value = mock_openai_models()

    foo_model = Model(identifier="foo", provider_resource_id="foo", provider_id="vllm-inference")

    await vllm_inference_adapter.register_model(foo_model)
    mock_openai_models_list.assert_called()


@pytest.mark.asyncio
async def test_old_vllm_tool_choice(vllm_inference_adapter):
    """
    Test that we set tool_choice to none when no tools are in use
    to support older versions of vLLM
    """
    mock_model = Model(identifier="mock-model", provider_resource_id="mock-model", provider_id="vllm-inference")
    vllm_inference_adapter.model_store.get_model.return_value = mock_model

    with patch.object(vllm_inference_adapter, "_nonstream_chat_completion") as mock_nonstream_completion:
        # No tools but auto tool choice
        await vllm_inference_adapter.chat_completion(
            "mock-model",
            [],
            stream=False,
            tools=None,
            tool_config=ToolConfig(tool_choice=ToolChoice.auto),
        )
        mock_nonstream_completion.assert_called()
        request = mock_nonstream_completion.call_args.args[0]
        # Ensure tool_choice gets converted to none for older vLLM versions
        assert request.tool_config.tool_choice == ToolChoice.none


@pytest.mark.asyncio
async def test_tool_call_delta_empty_tool_call_buf():
    """
    Test that we don't generate extra chunks when processing a
    tool call response that didn't call any tools. Previously we would
    emit chunks with spurious ToolCallParseStatus.succeeded or
    ToolCallParseStatus.failed when processing chunks that didn't
    actually make any tool calls.
    """

    async def mock_stream():
        delta = OpenAIChoiceDelta(content="", tool_calls=None)
        choices = [OpenAIChoice(delta=delta, finish_reason="stop", index=0)]
        mock_chunk = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=choices,
        )
        for chunk in [mock_chunk]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 1
    assert chunks[0].event.stop_reason == StopReason.end_of_turn


@pytest.mark.asyncio
async def test_process_vllm_chat_completion_stream_response_no_choices():
    """
    Test that we don't error out when vLLM returns no choices for a
    completion request. This can happen when there's an error thrown
    in vLLM for example.
    """

    async def mock_stream():
        choices = []
        mock_chunk = OpenAIChatCompletionChunk(
            id="chunk-1",
            created=1,
            model="foo",
            object="chat.completion.chunk",
            choices=choices,
        )
        for chunk in [mock_chunk]:
            yield chunk

    chunks = [chunk async for chunk in _process_vllm_chat_completion_stream_response(mock_stream())]
    assert len(chunks) == 0


def test_chat_completion_doesnt_block_event_loop(caplog):
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    caplog.set_level(logging.WARNING)

    # Log when event loop is blocked for more than 200ms
    loop.slow_callback_duration = 0.5
    # Sleep for 500ms in our delayed http response
    sleep_time = 0.5

    mock_model = Model(identifier="mock-model", provider_resource_id="mock-model", provider_id="vllm-inference")
    mock_response = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1,
        "modle": "mock-model",
        "choices": [
            {
                "message": {"content": ""},
                "logprobs": None,
                "finish_reason": "stop",
                "index": 0,
            }
        ],
    }

    async def do_chat_completion():
        await inference_adapter.chat_completion(
            "mock-model",
            [],
            stream=False,
            tools=None,
            tool_config=ToolConfig(tool_choice=ToolChoice.auto),
        )

    with MockInferenceAdapterWithSleep(sleep_time, mock_response) as inference_adapter:
        inference_adapter.model_store = AsyncMock()
        inference_adapter.model_store.get_model.return_value = mock_model
        loop.run_until_complete(inference_adapter.initialize())

        # Clear the logs so far and run the actual chat completion we care about
        caplog.clear()
        loop.run_until_complete(do_chat_completion())

    # Ensure we don't have any asyncio warnings in the captured log
    # records from our chat completion call. A message gets logged
    # here any time we exceed the slow_callback_duration configured
    # above.
    asyncio_warnings = [record.message for record in caplog.records if record.name == "asyncio"]
    assert not asyncio_warnings
