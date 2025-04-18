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
        "openai:responses:non_streaming_01",
        "openai:responses:non_streaming_02",
    ],
)
def test_basic_non_streaming(openai_client, client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = openai_client.responses.create(
        model=text_model_id,
        input=question,
        stream=False,
    )
    output_text = response.output_text.lower().strip()
    assert len(output_text) > 0
    assert expected.lower() in output_text

    retrieved_response = openai_client.responses.retrieve(response_id=response.id)
    assert retrieved_response.output_text == response.output_text

    next_response = openai_client.responses.create(
        model=text_model_id, input="Repeat your previous response in all caps.", previous_response_id=response.id
    )
    next_output_text = next_response.output_text.strip()
    assert expected.upper() in next_output_text


@pytest.mark.parametrize(
    "test_case",
    [
        "openai:responses:streaming_01",
        "openai:responses:streaming_02",
    ],
)
def test_basic_streaming(openai_client, client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = openai_client.responses.create(
        model=text_model_id,
        input=question,
        stream=True,
        timeout=120,  # Increase timeout to 2 minutes for large conversation history
    )
    streamed_content = []
    response_id = ""
    for chunk in response:
        response_id = chunk.response.id
        streamed_content.append(chunk.response.output_text.strip())

    assert len(streamed_content) > 0
    assert expected.lower() in "".join(streamed_content).lower()

    retrieved_response = openai_client.responses.retrieve(response_id=response_id)
    assert retrieved_response.output_text == "".join(streamed_content)

    next_response = openai_client.responses.create(
        model=text_model_id,
        input="Repeat your previous response in all caps.",
        previous_response_id=response_id,
        stream=True,
    )
    next_streamed_content = []
    for chunk in next_response:
        next_streamed_content.append(chunk.response.output_text.strip())
    assert expected.upper() in "".join(next_streamed_content)
