# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import httpx
import openai
import pytest

from llama_stack import LlamaStackAsLibraryClient
from llama_stack.distribution.datatypes import AuthenticationRequiredError
from tests.common.mcp import make_mcp_server
from tests.verifications.openai_api.fixtures.fixtures import (
    case_id_generator,
    get_base_test_name,
    should_skip_test,
)
from tests.verifications.openai_api.fixtures.load import load_test_cases

responses_test_cases = load_test_cases("responses")


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_basic"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_basic(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.responses.create(
        model=model,
        input=case["input"],
        stream=False,
    )
    output_text = response.output_text.lower().strip()
    assert len(output_text) > 0
    assert case["output"].lower() in output_text

    retrieved_response = openai_client.responses.retrieve(response_id=response.id)
    assert retrieved_response.output_text == response.output_text

    next_response = openai_client.responses.create(
        model=model, input="Repeat your previous response in all caps.", previous_response_id=response.id
    )
    next_output_text = next_response.output_text.strip()
    assert case["output"].upper() in next_output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_basic"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_streaming_basic(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    import time

    response = openai_client.responses.create(
        model=model,
        input=case["input"],
        stream=True,
    )

    # Track events and timing to verify proper streaming
    events = []
    event_times = []
    response_id = ""

    start_time = time.time()

    for chunk in response:
        current_time = time.time()
        event_times.append(current_time - start_time)
        events.append(chunk)

        if chunk.type == "response.created":
            # Verify response.created is emitted first and immediately
            assert len(events) == 1, "response.created should be the first event"
            assert event_times[0] < 0.1, "response.created should be emitted immediately"
            assert chunk.response.status == "in_progress"
            response_id = chunk.response.id

        elif chunk.type == "response.completed":
            # Verify response.completed comes after response.created
            assert len(events) >= 2, "response.completed should come after response.created"
            assert chunk.response.status == "completed"
            assert chunk.response.id == response_id, "Response ID should be consistent"

            # Verify content quality
            output_text = chunk.response.output_text.lower().strip()
            assert len(output_text) > 0, "Response should have content"
            assert case["output"].lower() in output_text, f"Expected '{case['output']}' in response"

    # Verify we got both required events
    event_types = [event.type for event in events]
    assert "response.created" in event_types, "Missing response.created event"
    assert "response.completed" in event_types, "Missing response.completed event"

    # Verify event order
    created_index = event_types.index("response.created")
    completed_index = event_types.index("response.completed")
    assert created_index < completed_index, "response.created should come before response.completed"

    # Verify stored response matches streamed response
    retrieved_response = openai_client.responses.retrieve(response_id=response_id)
    final_event = events[-1]
    assert retrieved_response.output_text == final_event.response.output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_basic"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_streaming_incremental_content(request, openai_client, model, provider, verification_config, case):
    """Test that streaming actually delivers content incrementally, not just at the end."""
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    import time

    response = openai_client.responses.create(
        model=model,
        input=case["input"],
        stream=True,
    )

    # Track all events and their content to verify incremental streaming
    events = []
    content_snapshots = []
    event_times = []

    start_time = time.time()

    for chunk in response:
        current_time = time.time()
        event_times.append(current_time - start_time)
        events.append(chunk)

        # Track content at each event based on event type
        if chunk.type == "response.output_text.delta":
            # For delta events, track the delta content
            content_snapshots.append(chunk.delta)
        elif hasattr(chunk, "response") and hasattr(chunk.response, "output_text"):
            # For response.created/completed events, track the full output_text
            content_snapshots.append(chunk.response.output_text)
        else:
            content_snapshots.append("")

    # Verify we have the expected events
    event_types = [event.type for event in events]
    assert "response.created" in event_types, "Missing response.created event"
    assert "response.completed" in event_types, "Missing response.completed event"

    # Check if we have incremental content updates
    created_index = event_types.index("response.created")
    completed_index = event_types.index("response.completed")

    # The key test: verify content progression
    created_content = content_snapshots[created_index]
    completed_content = content_snapshots[completed_index]

    # Verify that response.created has empty or minimal content
    assert len(created_content) == 0, f"response.created should have empty content, got: {repr(created_content[:100])}"

    # Verify that response.completed has the full content
    assert len(completed_content) > 0, "response.completed should have content"
    assert case["output"].lower() in completed_content.lower(), f"Expected '{case['output']}' in final content"

    # Check for true incremental streaming by looking for delta events
    delta_events = [i for i, event_type in enumerate(event_types) if event_type == "response.output_text.delta"]

    # Assert that we have delta events (true incremental streaming)
    assert len(delta_events) > 0, "Expected delta events for true incremental streaming, but found none"

    # Verify delta events have content and accumulate to final content
    delta_content_total = ""
    non_empty_deltas = 0

    for delta_idx in delta_events:
        delta_content = content_snapshots[delta_idx]
        if delta_content:
            delta_content_total += delta_content
            non_empty_deltas += 1

    # Assert that we have meaningful delta content
    assert non_empty_deltas > 0, "Delta events found but none contain content"
    assert len(delta_content_total) > 0, "Delta events found but total delta content is empty"

    # Verify that the accumulated delta content matches the final content
    assert delta_content_total.strip() == completed_content.strip(), (
        f"Delta content '{delta_content_total}' should match final content '{completed_content}'"
    )

    # Verify timing: delta events should come between created and completed
    for delta_idx in delta_events:
        assert created_index < delta_idx < completed_index, (
            f"Delta event at index {delta_idx} should be between created ({created_index}) and completed ({completed_index})"
        )


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_multi_turn"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_multi_turn(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    previous_response_id = None
    for turn in case["turns"]:
        response = openai_client.responses.create(
            model=model,
            input=turn["input"],
            previous_response_id=previous_response_id,
            tools=turn["tools"] if "tools" in turn else None,
        )
        previous_response_id = response.id
        output_text = response.output_text.lower()
        assert turn["output"].lower() in output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_web_search"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_web_search(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.responses.create(
        model=model,
        input=case["input"],
        tools=case["tools"],
        stream=False,
    )
    assert len(response.output) > 1
    assert response.output[0].type == "web_search_call"
    assert response.output[0].status == "completed"
    assert response.output[1].type == "message"
    assert response.output[1].status == "completed"
    assert response.output[1].role == "assistant"
    assert len(response.output[1].content) > 0
    assert case["output"].lower() in response.output_text.lower().strip()


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_mcp_tool"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_mcp_tool(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    with make_mcp_server() as mcp_server_info:
        tools = case["tools"]
        for tool in tools:
            if tool["type"] == "mcp":
                tool["server_url"] = mcp_server_info["server_url"]

        response = openai_client.responses.create(
            model=model,
            input=case["input"],
            tools=tools,
            stream=False,
        )
        assert len(response.output) >= 3
        list_tools = response.output[0]
        assert list_tools.type == "mcp_list_tools"
        assert list_tools.server_label == "localmcp"
        assert len(list_tools.tools) == 2
        assert {t["name"] for t in list_tools.tools} == {"get_boiling_point", "greet_everyone"}

        call = response.output[1]
        assert call.type == "mcp_call"
        assert call.name == "get_boiling_point"
        assert json.loads(call.arguments) == {"liquid_name": "polyjuice", "celcius": True}
        assert call.error is None
        assert "-100" in call.output

        message = response.output[2]
        text_content = message.content[0].text
        assert "boiling point" in text_content.lower()

    with make_mcp_server(required_auth_token="test-token") as mcp_server_info:
        tools = case["tools"]
        for tool in tools:
            if tool["type"] == "mcp":
                tool["server_url"] = mcp_server_info["server_url"]

        exc_type = (
            AuthenticationRequiredError
            if isinstance(openai_client, LlamaStackAsLibraryClient)
            else (httpx.HTTPStatusError, openai.AuthenticationError)
        )
        with pytest.raises(exc_type):
            openai_client.responses.create(
                model=model,
                input=case["input"],
                tools=tools,
                stream=False,
            )

        for tool in tools:
            if tool["type"] == "mcp":
                tool["server_url"] = mcp_server_info["server_url"]
                tool["headers"] = {"Authorization": "Bearer test-token"}

        response = openai_client.responses.create(
            model=model,
            input=case["input"],
            tools=tools,
            stream=False,
        )
        assert len(response.output) >= 3


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_custom_tool"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_custom_tool(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.responses.create(
        model=model,
        input=case["input"],
        tools=case["tools"],
        stream=False,
    )
    assert len(response.output) == 1
    assert response.output[0].type == "function_call"
    assert response.output[0].status == "completed"
    assert response.output[0].name == "get_weather"


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_image"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_image(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    response = openai_client.responses.create(
        model=model,
        input=case["input"],
        stream=False,
    )
    output_text = response.output_text.lower()
    assert case["output"].lower() in output_text


@pytest.mark.parametrize(
    "case",
    responses_test_cases["test_response_multi_turn_image"]["test_params"]["case"],
    ids=case_id_generator,
)
def test_response_non_streaming_multi_turn_image(request, openai_client, model, provider, verification_config, case):
    test_name_base = get_base_test_name(request)
    if should_skip_test(verification_config, provider, model, test_name_base):
        pytest.skip(f"Skipping {test_name_base} for model {model} on provider {provider} based on config.")

    previous_response_id = None
    for turn in case["turns"]:
        response = openai_client.responses.create(
            model=model,
            input=turn["input"],
            previous_response_id=previous_response_id,
            tools=turn["tools"] if "tools" in turn else None,
        )
        previous_response_id = response.id
        output_text = response.output_text.lower()
        assert turn["output"].lower() in output_text
