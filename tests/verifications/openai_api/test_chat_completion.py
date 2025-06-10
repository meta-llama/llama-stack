# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from openai import APIError
from pydantic import BaseModel

from tests.verifications.openai_api.fixtures.fixtures import (
    case_id_generator,
    get_base_test_name,
    should_skip_test,
)
from tests.verifications.openai_api.fixtures.load import load_test_cases

chat_completion_test_cases = load_test_cases("chat_completion")

THIS_DIR = Path(__file__).parent


@pytest.fixture
def multi_image_data():
    files = [
        THIS_DIR / "fixtures/images/vision_test_1.jpg",
        THIS_DIR / "fixtures/images/vision_test_2.jpg",
        THIS_DIR / "fixtures/images/vision_test_3.jpg",
    ]
    encoded_files = []
    for file in files:
        with open(file, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            encoded_files.append(f"data:image/jpeg;base64,{base64_data}")
    return encoded_files


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
    chat_completion_test_cases["test_chat_input_validation"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_non_streaming_error_handling(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    with pytest.raises(APIError) as e:
        openai_client.chat.completions.create(
            model=model,
            messages=case["input"]["messages"],
            stream=False,
            tool_choice=case["input"]["tool_choice"] if "tool_choice" in case["input"] else None,
            tools=case["input"]["tools"] if "tools" in case["input"] else None,
        )
    assert case["output"]["error"]["status_code"] == e.value.status_code


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_chat_input_validation"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_chat_streaming_error_handling(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    with pytest.raises(APIError) as e:
        response = openai_client.chat.completions.create(
            model=model,
            messages=case["input"]["messages"],
            stream=True,
            tool_choice=case["input"]["tool_choice"] if "tool_choice" in case["input"] else None,
            tools=case["input"]["tools"] if "tools" in case["input"] else None,
        )
        for _chunk in response:
            pass
    assert str(case["output"]["error"]["status_code"]) in e.value.message


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

    _, tool_calls_buffer = _accumulate_streaming_tool_calls(stream)
    assert len(tool_calls_buffer) == 1
    for call in tool_calls_buffer:
        assert len(call["id"]) > 0
        function = call["function"]
        assert function["name"] == "get_weather"

        args_dict = json.loads(function["arguments"])
        assert "san francisco" in args_dict["location"].lower()


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_tool_calling"]["test_params"]["case"],  # Reusing existing case for now
    ids=case_id_generator,
)
def test_chat_non_streaming_tool_choice_required(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        tools=case["input"]["tools"],
        tool_choice="required",  # Force tool call
        stream=False,
    )

    assert response.choices[0].message.role == "assistant"
    assert len(response.choices[0].message.tool_calls) > 0, "Expected tool call when tool_choice='required'"
    expected_tool_name = case["input"]["tools"][0]["function"]["name"]
    assert response.choices[0].message.tool_calls[0].function.name == expected_tool_name


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_tool_calling"]["test_params"]["case"],  # Reusing existing case for now
    ids=case_id_generator,
)
def test_chat_streaming_tool_choice_required(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    stream = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        tools=case["input"]["tools"],
        tool_choice="required",  # Force tool call
        stream=True,
    )

    _, tool_calls_buffer = _accumulate_streaming_tool_calls(stream)

    assert len(tool_calls_buffer) > 0, "Expected tool call when tool_choice='required'"
    expected_tool_name = case["input"]["tools"][0]["function"]["name"]
    assert any(call["function"]["name"] == expected_tool_name for call in tool_calls_buffer), (
        f"Expected tool call '{expected_tool_name}' not found in stream"
    )


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_tool_calling"]["test_params"]["case"],  # Reusing existing case for now
    ids=case_id_generator,
)
def test_chat_non_streaming_tool_choice_none(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        tools=case["input"]["tools"],
        tool_choice="none",
        stream=False,
    )

    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.tool_calls is None, "Expected no tool calls when tool_choice='none'"
    assert response.choices[0].message.content is not None, "Expected content when tool_choice='none'"


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases["test_tool_calling"]["test_params"]["case"],  # Reusing existing case for now
    ids=case_id_generator,
)
def test_chat_streaming_tool_choice_none(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    stream = openai_client.chat.completions.create(
        model=model,
        messages=case["input"]["messages"],
        tools=case["input"]["tools"],
        tool_choice="none",
        stream=True,
    )

    content = ""
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
        assert not delta.tool_calls, "Expected no tool call chunks when tool_choice='none'"

    assert len(content) > 0, "Expected content when tool_choice='none'"


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases.get("test_chat_multi_turn_tool_calling", {}).get("test_params", {}).get("case", []),
    ids=case_id_generator,
)
def test_chat_non_streaming_multi_turn_tool_calling(request, openai_client, model, provider, verification_config, case):
    """
    Test cases for multi-turn tool calling.
    Tool calls are asserted.
    Tool responses are provided in the test case.
    Final response is asserted.
    """

    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    # Create a copy of the messages list to avoid modifying the original
    messages = []
    tools = case["input"]["tools"]
    # Use deepcopy to prevent modification across runs/parametrization
    expected_results = copy.deepcopy(case["expected"])
    tool_responses = copy.deepcopy(case.get("tool_responses", []))
    input_messages_turns = copy.deepcopy(case["input"]["messages"])

    # keep going until either
    # 1. we have messages to test in multi-turn
    # 2. no messages but last message is tool response
    while len(input_messages_turns) > 0 or (len(messages) > 0 and messages[-1]["role"] == "tool"):
        # do not take new messages if last message is tool response
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            new_messages = input_messages_turns.pop(0)
            # Ensure new_messages is a list of message objects
            if isinstance(new_messages, list):
                messages.extend(new_messages)
            else:
                # If it's a single message object, add it directly
                messages.append(new_messages)

        # --- API Call ---
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=False,
        )

        # --- Process Response ---
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump(exclude_unset=True))

        assert assistant_message.role == "assistant"

        # Get the expected result data
        expected = expected_results.pop(0)
        num_tool_calls = expected["num_tool_calls"]

        # --- Assertions based on expected result ---
        assert len(assistant_message.tool_calls or []) == num_tool_calls, (
            f"Expected {num_tool_calls} tool calls, but got {len(assistant_message.tool_calls or [])}"
        )

        if num_tool_calls > 0:
            tool_call = assistant_message.tool_calls[0]
            assert tool_call.function.name == expected["tool_name"], (
                f"Expected tool '{expected['tool_name']}', got '{tool_call.function.name}'"
            )
            # Parse the JSON string arguments before comparing
            actual_arguments = json.loads(tool_call.function.arguments)
            assert actual_arguments == expected["tool_arguments"], (
                f"Expected arguments '{expected['tool_arguments']}', got '{actual_arguments}'"
            )

            # Prepare and append the tool response for the next turn
            tool_response = tool_responses.pop(0)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_response["response"],
                }
            )
        else:
            assert assistant_message.content is not None, "Expected content, but none received."
            expected_answers = expected["answer"]  # This is now a list
            content_lower = assistant_message.content.lower()
            assert any(ans.lower() in content_lower for ans in expected_answers), (
                f"Expected one of {expected_answers} in content, but got: '{assistant_message.content}'"
            )


@pytest.mark.parametrize(
    "case",
    chat_completion_test_cases.get("test_chat_multi_turn_tool_calling", {}).get("test_params", {}).get("case", []),
    ids=case_id_generator,
)
def test_chat_streaming_multi_turn_tool_calling(request, openai_client, model, provider, verification_config, case):
    """ """
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    messages = []
    tools = case["input"]["tools"]
    expected_results = copy.deepcopy(case["expected"])
    tool_responses = copy.deepcopy(case.get("tool_responses", []))
    input_messages_turns = copy.deepcopy(case["input"]["messages"])

    while len(input_messages_turns) > 0 or (len(messages) > 0 and messages[-1]["role"] == "tool"):
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            new_messages = input_messages_turns.pop(0)
            if isinstance(new_messages, list):
                messages.extend(new_messages)
            else:
                messages.append(new_messages)

        # --- API Call (Streaming) ---
        stream = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            stream=True,
        )

        # --- Process Stream ---
        accumulated_content, accumulated_tool_calls = _accumulate_streaming_tool_calls(stream)

        # --- Construct Assistant Message for History ---
        assistant_message_dict = {"role": "assistant"}
        if accumulated_content:
            assistant_message_dict["content"] = accumulated_content
        if accumulated_tool_calls:
            assistant_message_dict["tool_calls"] = accumulated_tool_calls

        messages.append(assistant_message_dict)

        # --- Assertions ---
        expected = expected_results.pop(0)
        num_tool_calls = expected["num_tool_calls"]

        assert len(accumulated_tool_calls or []) == num_tool_calls, (
            f"Expected {num_tool_calls} tool calls, but got {len(accumulated_tool_calls or [])}"
        )

        if num_tool_calls > 0:
            # Use the first accumulated tool call for assertion
            tool_call = accumulated_tool_calls[0]
            assert tool_call["function"]["name"] == expected["tool_name"], (
                f"Expected tool '{expected['tool_name']}', got '{tool_call['function']['name']}'"
            )
            # Parse the accumulated arguments string for comparison
            actual_arguments = json.loads(tool_call["function"]["arguments"])
            assert actual_arguments == expected["tool_arguments"], (
                f"Expected arguments '{expected['tool_arguments']}', got '{actual_arguments}'"
            )

            # Prepare and append the tool response for the next turn
            tool_response = tool_responses.pop(0)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_response["response"],
                }
            )
        else:
            assert accumulated_content is not None and accumulated_content != "", "Expected content, but none received."
            expected_answers = expected["answer"]
            content_lower = accumulated_content.lower()
            assert any(ans.lower() in content_lower for ans in expected_answers), (
                f"Expected one of {expected_answers} in content, but got: '{accumulated_content}'"
            )


@pytest.mark.parametrize("stream", [False, True], ids=["stream=False", "stream=True"])
def test_chat_multi_turn_multiple_images(
    request, openai_client, model, provider, verification_config, multi_image_data, stream
):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    messages_turn1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": multi_image_data[0],
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": multi_image_data[1],
                    },
                },
                {
                    "type": "text",
                    "text": "What furniture is in the first image that is not in the second image?",
                },
            ],
        },
    ]

    # First API call
    response1 = openai_client.chat.completions.create(
        model=model,
        messages=messages_turn1,
        stream=stream,
    )
    if stream:
        message_content1 = ""
        for chunk in response1:
            message_content1 += chunk.choices[0].delta.content or ""
    else:
        message_content1 = response1.choices[0].message.content
    assert len(message_content1) > 0
    assert any(expected in message_content1.lower().strip() for expected in {"chair", "table"}), message_content1

    # Prepare messages for the second turn
    messages_turn2 = messages_turn1 + [
        {"role": "assistant", "content": message_content1},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": multi_image_data[2],
                    },
                },
                {"type": "text", "text": "What is in this image that is also in the first image?"},
            ],
        },
    ]

    # Second API call
    response2 = openai_client.chat.completions.create(
        model=model,
        messages=messages_turn2,
        stream=stream,
    )
    if stream:
        message_content2 = ""
        for chunk in response2:
            message_content2 += chunk.choices[0].delta.content or ""
    else:
        message_content2 = response2.choices[0].message.content
    assert len(message_content2) > 0
    assert any(expected in message_content2.lower().strip() for expected in {"bed"}), message_content2


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


def _accumulate_streaming_tool_calls(stream):
    """Accumulates tool calls and content from a streaming ChatCompletion response."""
    tool_calls_buffer = {}
    current_id = None
    full_content = ""  # Initialize content accumulator
    # Process streaming chunks
    for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        # Accumulate content
        if delta.content:
            full_content += delta.content

        if delta.tool_calls is None:
            continue

        for tool_call_delta in delta.tool_calls:
            if tool_call_delta.id:
                current_id = tool_call_delta.id
            call_id = current_id
            # Skip if no ID seen yet for this tool call delta
            if not call_id:
                continue
            func_delta = tool_call_delta.function

            if call_id not in tool_calls_buffer:
                tool_calls_buffer[call_id] = {
                    "id": call_id,
                    "type": "function",  # Assume function type
                    "function": {"name": None, "arguments": ""},  # Nested structure
                }

            # Accumulate name and arguments into the nested function dict
            if func_delta:
                if func_delta.name:
                    tool_calls_buffer[call_id]["function"]["name"] = func_delta.name
                if func_delta.arguments:
                    tool_calls_buffer[call_id]["function"]["arguments"] += func_delta.arguments

    # Return content and tool calls as a list
    return full_content, list(tool_calls_buffer.values())
