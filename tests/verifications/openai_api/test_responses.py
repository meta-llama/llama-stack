# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

import httpx
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

    response = openai_client.responses.create(
        model=model,
        input=case["input"],
        stream=True,
    )
    streamed_content = []
    response_id = ""
    for chunk in response:
        if chunk.type == "response.completed":
            response_id = chunk.response.id
            streamed_content.append(chunk.response.output_text.strip())

    assert len(streamed_content) > 0
    assert case["output"].lower() in "".join(streamed_content).lower()

    retrieved_response = openai_client.responses.retrieve(response_id=response_id)
    assert retrieved_response.output_text == "".join(streamed_content)


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
            else httpx.HTTPStatusError
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
