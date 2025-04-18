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
