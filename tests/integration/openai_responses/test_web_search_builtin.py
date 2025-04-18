# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from ..test_cases.test_case import TestCase


@pytest.mark.parametrize(
    "test_case",
    [
        "openai:responses:tools_web_search_01",
    ],
)
def test_web_search_non_streaming(openai_client, client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)
    input = tc["input"]
    expected = tc["expected"]
    tools = tc["tools"]

    response = openai_client.responses.create(
        model=text_model_id,
        input=input,
        tools=tools,
        stream=False,
    )
    assert len(response.output) > 1
    assert response.output[0].type == "web_search_call"
    assert response.output[0].status == "completed"
    assert response.output[1].type == "message"
    assert response.output[1].status == "completed"
    assert response.output[1].role == "assistant"
    assert len(response.output[1].content) > 0
    assert expected.lower() in response.output_text.lower().strip()


def test_input_image_non_streaming(openai_client, vision_model_id):
    supported_models = ["llama-4", "gpt-4o", "llama4"]
    if not any(model in vision_model_id.lower() for model in supported_models):
        pytest.skip(f"Skip for non-supported model: {vision_model_id}")

    response = openai_client.with_options(max_retries=0).responses.create(
        model=vision_model_id,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Identify the type of animal in this image.",
                    },
                    {
                        "type": "input_image",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/f/f7/Llamas%2C_Vernagt-Stausee%2C_Italy.jpg",
                    },
                ],
            }
        ],
    )
    output_text = response.output_text.lower()
    assert "llama" in output_text


def test_multi_turn_web_search_from_image_non_streaming(openai_client, vision_model_id):
    supported_models = ["llama-4", "gpt-4o", "llama4"]
    if not any(model in vision_model_id.lower() for model in supported_models):
        pytest.skip(f"Skip for non-supported model: {vision_model_id}")

    response = openai_client.with_options(max_retries=0).responses.create(
        model=vision_model_id,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract a single search keyword that represents the type of animal in this image.",
                    },
                    {
                        "type": "input_image",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/f/f7/Llamas%2C_Vernagt-Stausee%2C_Italy.jpg",
                    },
                ],
            }
        ],
    )
    output_text = response.output_text.lower()
    assert "llama" in output_text

    search_response = openai_client.with_options(max_retries=0).responses.create(
        model=vision_model_id,
        input="Search the web using the search tool for those keywords plus the words 'maverick' and 'scout' and summarize the results.",
        previous_response_id=response.id,
        tools=[{"type": "web_search"}],
    )
    output_text = search_response.output_text.lower()
    assert "model" in output_text
