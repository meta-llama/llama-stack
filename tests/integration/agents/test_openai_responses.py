# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from urllib.parse import urljoin

import pytest
import requests
from openai import OpenAI

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient


@pytest.fixture
def openai_client(client_with_models):
    base_url = f"{client_with_models.base_url}/v1/openai/v1"
    return OpenAI(base_url=base_url, api_key="bar")


@pytest.mark.parametrize(
    "stream",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    "tools",
    [
        [],
        [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city to get the weather for"},
                    },
                },
            }
        ],
    ],
)
def test_responses_store(openai_client, client_with_models, text_model_id, stream, tools):
    if isinstance(client_with_models, LlamaStackAsLibraryClient):
        pytest.skip("OpenAI responses are not supported when testing with library client yet.")

    client = openai_client
    message = "What's the weather in Tokyo?" + (
        " YOU MUST USE THE get_weather function to get the weather." if tools else ""
    )
    response = client.responses.create(
        model=text_model_id,
        input=[
            {
                "role": "user",
                "content": message,
            }
        ],
        stream=stream,
        tools=tools,
    )
    if stream:
        # accumulate the streamed content
        content = ""
        response_id = None
        for chunk in response:
            if response_id is None:
                response_id = chunk.response.id
            if not tools:
                if chunk.type == "response.completed":
                    response_id = chunk.response.id
                    content = chunk.response.output[0].content[0].text
    else:
        response_id = response.id
        if not tools:
            content = response.output[0].content[0].text

    # list responses is not available in the SDK
    url = urljoin(str(client.base_url), "responses")
    response = requests.get(url, headers={"Authorization": f"Bearer {client.api_key}"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert response_id in [r["id"] for r in data]

    # test retrieve response
    retrieved_response = client.responses.retrieve(response_id)
    assert retrieved_response.id == response_id
    assert retrieved_response.model == text_model_id
    if tools:
        assert retrieved_response.output[0].type == "function_call"
    else:
        assert retrieved_response.output[0].content[0].text == content
