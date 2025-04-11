# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import re
from typing import Any

import pytest
from pydantic import BaseModel

from tests.verifications.openai_api.fixtures.fixtures import _load_all_verification_configs
from tests.verifications.openai_api.fixtures.load import load_test_cases

chat_completion_test_cases = load_test_cases("chat_completion")


def case_id_generator(case):
    """Generate a test ID from the case's 'case_id' field, or use a default."""
    case_id = case.get("case_id")
    if isinstance(case_id, (str, int)):
        return re.sub(r"\\W|^(?=\\d)", "_", str(case_id))
    return None


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on the selected provider and config."""
    if "model" in metafunc.fixturenames:
        provider = metafunc.config.getoption("provider")
        if not provider:
            print("Warning: --provider not specified. Skipping model parametrization.")
            metafunc.parametrize("model", [])
            return

        try:
            config_data = _load_all_verification_configs()
        except (FileNotFoundError, IOError) as e:
            print(f"ERROR loading verification configs: {e}")
            config_data = {"providers": {}}

        provider_config = config_data.get("providers", {}).get(provider)
        if provider_config:
            models = provider_config.get("models", [])
            if models:
                metafunc.parametrize("model", models)
            else:
                print(f"Warning: No models found for provider '{provider}' in config.")
                metafunc.parametrize("model", [])  # Parametrize empty if no models found
        else:
            print(f"Warning: Provider '{provider}' not found in config. No models parametrized.")
            metafunc.parametrize("model", [])  # Parametrize empty if provider not found


def should_skip_test(verification_config, provider, model, test_name_base):
    """Check if a test should be skipped based on config exclusions."""
    provider_config = verification_config.get("providers", {}).get(provider)
    if not provider_config:
        return False  # No config for provider, don't skip

    exclusions = provider_config.get("test_exclusions", {}).get(model, [])
    return test_name_base in exclusions


# Helper to get the base test name from the request object
def get_base_test_name(request):
    return request.node.originalname


# --- Test Functions ---


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_chat_basic"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_non_streaming_basic(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        stream=False,
    )
    assert response.choices[0].message.role == "assistant"
    assert case["output"].lower() in response.choices[0].message.content.lower()


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_chat_basic"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_streaming_basic(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        stream=True,
    )
    content = ""
    for chunk in response:
        content += chunk.choices[0].delta.content or ""

    # TODO: add detailed type validation

    assert case["output"].lower() in content.lower()


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_chat_image"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_non_streaming_image(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        stream=False,
    )
    assert response.choices[0].message.role == "assistant"
    assert case["output"].lower() in response.choices[0].message.content.lower()


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_chat_image"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_streaming_image(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        stream=True,
    )
    content = ""
    for chunk in response:
        content += chunk.choices[0].delta.content or ""

    # TODO: add detailed type validation

    assert case["output"].lower() in content.lower()


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_chat_structured_output"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_non_streaming_structured_output(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        response_format=case["input"]["response_format"],
        stream=False,
    )

    assert response.choices[0].message.role == "assistant"
    maybe_json_content = response.choices[0].message.content

    validate_structured_output(maybe_json_content, case["output"])


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_chat_structured_output"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_streaming_structured_output(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        response_format=case["input"]["response_format"],
        stream=True,
    )
    maybe_json_content = ""
    for chunk in response:
        maybe_json_content += chunk.choices[0].delta.content or ""
    validate_structured_output(maybe_json_content, case["output"])


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_tool_calling"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_non_streaming_tool_calling(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        tools=case["input"]["tools"],
        stream=False,
    )

    assert response.choices[0].message.role == "assistant"
    assert len(response.choices[0].message.tool_calls) > 0
    assert case["output"] == "get_weather_tool_call"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
    # TODO: add detailed type validation


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_tool_calling"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_streaming_tool_calling(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    stream = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        tools=case["input"]["tools"],
        stream=True,
    )

    # Accumulate partial tool_calls here
    tool_calls_buffer = {}
    current_id = None
    # Process streaming chunks
    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        if delta.tool_calls is None:
            continue

        for tool_call_delta in delta.tool_calls:
            if tool_call_delta.id:
                current_id = tool_call_delta.id
            call_id = current_id
            func_delta = tool_call_delta.function

            if call_id not in tool_calls_buffer:
                tool_calls_buffer[call_id] = {
                    "id": call_id,
                    "type": tool_call_delta.type,
                    "name": func_delta.name,
                    "arguments": "",
                }

            if func_delta.arguments:
                tool_calls_buffer[call_id]["arguments"] += func_delta.arguments

    assert len(tool_calls_buffer) == 1
    for call in tool_calls_buffer.values():
        assert len(call["id"]) > 0
        assert call["name"] == "get_weather"

        args_dict = json.loads(call["arguments"])
        assert "san francisco" in args_dict["location"].lower()


# --- Helper functions (structured output validation) ---


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
