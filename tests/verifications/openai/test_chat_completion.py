# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

import pytest
from pydantic import BaseModel

from tests.verifications.openai.fixtures.load import load_test_cases

chat_completion_test_cases = load_test_cases("chat_completion")


@pytest.fixture
def correct_model_name(model, provider, providers_model_mapping):
    """Return the provider-specific model name based on the generic model name."""
    mapping = providers_model_mapping[provider]
    if model not in mapping:
        pytest.skip(f"Provider {provider} does not support model {model}")
    return mapping[model]


@pytest.mark.parametrize("model", chat_completion_test_cases["test_chat_basic"]["test_params"]["model"])
@pytest.mark.parametrize(
    "input_output",
    chat_completion_test_cases["test_chat_basic"]["test_params"]["input_output"],
)
def test_chat_non_streaming_basic(openai_client, input_output, correct_model_name):
    response = openai_client.chat.completions.create(
        model=correct_model_name,
        messages=input_output["input"]["messages"],
        stream=False,
    )
    assert response.choices[0].message.role == "assistant"
    assert input_output["output"].lower() in response.choices[0].message.content.lower()


@pytest.mark.parametrize("model", chat_completion_test_cases["test_chat_basic"]["test_params"]["model"])
@pytest.mark.parametrize(
    "input_output",
    chat_completion_test_cases["test_chat_basic"]["test_params"]["input_output"],
)
def test_chat_streaming_basic(openai_client, input_output, correct_model_name):
    response = openai_client.chat.completions.create(
        model=correct_model_name,
        messages=input_output["input"]["messages"],
        stream=True,
    )
    content = ""
    for chunk in response:
        content += chunk.choices[0].delta.content or ""

    # TODO: add detailed type validation

    assert input_output["output"].lower() in content.lower()


@pytest.mark.parametrize("model", chat_completion_test_cases["test_chat_image"]["test_params"]["model"])
@pytest.mark.parametrize(
    "input_output",
    chat_completion_test_cases["test_chat_image"]["test_params"]["input_output"],
)
def test_chat_non_streaming_image(openai_client, input_output, correct_model_name):
    response = openai_client.chat.completions.create(
        model=correct_model_name,
        messages=input_output["input"]["messages"],
        stream=False,
    )
    assert response.choices[0].message.role == "assistant"
    assert input_output["output"].lower() in response.choices[0].message.content.lower()


@pytest.mark.parametrize("model", chat_completion_test_cases["test_chat_image"]["test_params"]["model"])
@pytest.mark.parametrize(
    "input_output",
    chat_completion_test_cases["test_chat_image"]["test_params"]["input_output"],
)
def test_chat_streaming_image(openai_client, input_output, correct_model_name):
    response = openai_client.chat.completions.create(
        model=correct_model_name,
        messages=input_output["input"]["messages"],
        stream=True,
    )
    content = ""
    for chunk in response:
        content += chunk.choices[0].delta.content or ""

    # TODO: add detailed type validation

    assert input_output["output"].lower() in content.lower()


@pytest.mark.parametrize(
    "model",
    chat_completion_test_cases["test_chat_structured_output"]["test_params"]["model"],
)
@pytest.mark.parametrize(
    "input_output",
    chat_completion_test_cases["test_chat_structured_output"]["test_params"]["input_output"],
)
def test_chat_non_streaming_structured_output(openai_client, input_output, correct_model_name):
    response = openai_client.chat.completions.create(
        model=correct_model_name,
        messages=input_output["input"]["messages"],
        response_format=input_output["input"]["response_format"],
        stream=False,
    )

    assert response.choices[0].message.role == "assistant"
    maybe_json_content = response.choices[0].message.content

    validate_structured_output(maybe_json_content, input_output["output"])


@pytest.mark.parametrize(
    "model",
    chat_completion_test_cases["test_chat_structured_output"]["test_params"]["model"],
)
@pytest.mark.parametrize(
    "input_output",
    chat_completion_test_cases["test_chat_structured_output"]["test_params"]["input_output"],
)
def test_chat_streaming_structured_output(openai_client, input_output, correct_model_name):
    response = openai_client.chat.completions.create(
        model=correct_model_name,
        messages=input_output["input"]["messages"],
        response_format=input_output["input"]["response_format"],
        stream=True,
    )
    maybe_json_content = ""
    for chunk in response:
        maybe_json_content += chunk.choices[0].delta.content or ""
    validate_structured_output(maybe_json_content, input_output["output"])


@pytest.mark.parametrize(
    "model",
    chat_completion_test_cases["test_tool_calling"]["test_params"]["model"],
)
@pytest.mark.parametrize(
    "input_output",
    chat_completion_test_cases["test_tool_calling"]["test_params"]["input_output"],
)
def test_chat_non_streaming_tool_calling(openai_client, input_output, correct_model_name):
    response = openai_client.chat.completions.create(
        model=correct_model_name,
        messages=input_output["input"]["messages"],
        tools=input_output["input"]["tools"],
        stream=False,
    )

    assert response.choices[0].message.role == "assistant"
    assert len(response.choices[0].message.tool_calls) > 0
    assert input_output["output"] == "get_weather_tool_call"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
    # TODO: add detailed type validation


def get_structured_output(maybe_json_content: str, schema_name: str) -> Any | None:
    if schema_name == "valid_calendar_event":

        class CalendarEvent(BaseModel):
            name: str
            date: str
            participants: list[str]

        try:
            calendar_event = CalendarEvent.model_validate_json(maybe_json_content)
            return calendar_event
        except Exception:
            return None
    elif schema_name == "valid_math_reasoning":

        class Step(BaseModel):
            explanation: str
            output: str

        class MathReasoning(BaseModel):
            steps: list[Step]
            final_answer: str

        try:
            math_reasoning = MathReasoning.model_validate_json(maybe_json_content)
            return math_reasoning
        except Exception:
            return None

    return None


def validate_structured_output(maybe_json_content: str, schema_name: str) -> None:
    structured_output = get_structured_output(maybe_json_content, schema_name)
    assert structured_output is not None
    if schema_name == "valid_calendar_event":
        assert structured_output.name is not None
        assert structured_output.date is not None
        assert len(structured_output.participants) == 2
    elif schema_name == "valid_math_reasoning":
        assert len(structured_output.final_answer) > 0
