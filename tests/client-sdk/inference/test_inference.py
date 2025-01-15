# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from pydantic import BaseModel


PROVIDER_TOOL_PROMPT_FORMAT = {
    "remote::ollama": "python_list",
    "remote::together": "json",
    "remote::fireworks": "json",
}


@pytest.fixture(scope="session")
def provider_tool_format(inference_provider_type):
    return (
        PROVIDER_TOOL_PROMPT_FORMAT[inference_provider_type]
        if inference_provider_type in PROVIDER_TOOL_PROMPT_FORMAT
        else None
    )


@pytest.fixture(scope="session")
def inference_provider_type(llama_stack_client):
    providers = llama_stack_client.providers.list()
    assert len(providers.inference) > 0
    return providers.inference[0]["provider_type"]


@pytest.fixture(scope="session")
def text_model_id(llama_stack_client):
    available_models = [
        model.identifier
        for model in llama_stack_client.models.list().data
        if model.identifier.startswith("meta-llama") and "405" not in model.identifier
    ]
    print(available_models)
    assert len(available_models) > 0
    return available_models[0]


@pytest.fixture(scope="session")
def vision_model_id(llama_stack_client):
    available_models = [
        model.identifier
        for model in llama_stack_client.models.list().data
        if "vision" in model.identifier.lower()
    ]
    if len(available_models) == 0:
        pytest.skip("No vision models available")

    return available_models[0]


@pytest.fixture
def get_weather_tool_definition():
    return {
        "tool_name": "get_weather",
        "description": "Get the current weather",
        "parameters": {
            "location": {
                "param_type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
        },
    }


def test_completion_non_streaming(llama_stack_client, text_model_id):
    response = llama_stack_client.inference.completion(
        content="Complete the sentence using one word: Roses are red, violets are ",
        stream=False,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 50,
        },
    )
    assert "blue" in response.content.lower().strip()


def test_completion_streaming(llama_stack_client, text_model_id):
    response = llama_stack_client.inference.completion(
        content="Complete the sentence using one word: Roses are red, violets are ",
        stream=True,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 50,
        },
    )
    streamed_content = [chunk.delta for chunk in response]
    assert "blue" in "".join(streamed_content).lower().strip()


def test_completion_log_probs_non_streaming(llama_stack_client, text_model_id):
    response = llama_stack_client.inference.completion(
        content="Complete the sentence: Micheael Jordan is born in ",
        stream=False,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 5,
        },
        logprobs={
            "top_k": 3,
        },
    )
    assert response.logprobs, "Logprobs should not be empty"
    assert 1 <= len(response.logprobs) <= 5
    assert all(len(logprob.logprobs_by_token) == 3 for logprob in response.logprobs)


def test_completion_log_probs_streaming(llama_stack_client, text_model_id):
    response = llama_stack_client.inference.completion(
        content="Complete the sentence: Micheael Jordan is born in ",
        stream=True,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 5,
        },
        logprobs={
            "top_k": 3,
        },
    )
    streamed_content = [chunk for chunk in response]
    for chunk in streamed_content:
        if chunk.delta:  # if there's a token, we expect logprobs
            assert chunk.logprobs, "Logprobs should not be empty"
            assert all(
                len(logprob.logprobs_by_token) == 3 for logprob in chunk.logprobs
            )
        else:  # no token, no logprobs
            assert not chunk.logprobs, "Logprobs should be empty"


def test_completion_structured_output(
    llama_stack_client, text_model_id, inference_provider_type
):
    user_input = """
    Michael Jordan was born in 1963. He played basketball for the Chicago Bulls. He retired in 2003.
    """

    class AnswerFormat(BaseModel):
        name: str
        year_born: str
        year_retired: str

    response = llama_stack_client.inference.completion(
        model_id=text_model_id,
        content=user_input,
        stream=False,
        sampling_params={
            "max_tokens": 50,
        },
        response_format={
            "type": "json_schema",
            "json_schema": AnswerFormat.model_json_schema(),
        },
    )
    answer = AnswerFormat.model_validate_json(response.content)
    assert answer.name == "Michael Jordan"
    assert answer.year_born == "1963"
    assert answer.year_retired == "2003"


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What are the names of planets in our solar system?", "Earth"),
        ("What are the names of the planets that have rings around them?", "Saturn"),
    ],
)
def test_text_chat_completion_non_streaming(
    llama_stack_client, text_model_id, question, expected
):
    response = llama_stack_client.inference.chat_completion(
        model_id=text_model_id,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        stream=False,
    )
    message_content = response.completion_message.content.lower().strip()
    assert len(message_content) > 0
    assert expected.lower() in message_content


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What's the name of the Sun in latin?", "Sol"),
        ("What is the name of the US captial?", "Washington"),
    ],
)
def test_text_chat_completion_streaming(
    llama_stack_client, text_model_id, question, expected
):
    response = llama_stack_client.inference.chat_completion(
        model_id=text_model_id,
        messages=[{"role": "user", "content": question}],
        stream=True,
    )
    streamed_content = [
        str(chunk.event.delta.text.lower().strip()) for chunk in response
    ]
    assert len(streamed_content) > 0
    assert expected.lower() in "".join(streamed_content)


def test_text_chat_completion_with_tool_calling_and_non_streaming(
    llama_stack_client, text_model_id, get_weather_tool_definition, provider_tool_format
):
    response = llama_stack_client.inference.chat_completion(
        model_id=text_model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like in San Francisco?"},
        ],
        tools=[get_weather_tool_definition],
        tool_choice="auto",
        tool_prompt_format=provider_tool_format,
        stream=False,
    )
    # No content is returned for the system message since we expect the
    # response to be a tool call
    assert response.completion_message.content == ""
    assert response.completion_message.role == "assistant"
    assert response.completion_message.stop_reason == "end_of_turn"

    assert len(response.completion_message.tool_calls) == 1
    assert response.completion_message.tool_calls[0].tool_name == "get_weather"
    assert response.completion_message.tool_calls[0].arguments == {
        "location": "San Francisco, CA"
    }


# Will extract streamed text and separate it from tool invocation content
# The returned tool inovcation content will be a string so it's easy to comapare with expected value
# e.g. "[get_weather, {'location': 'San Francisco, CA'}]"
def extract_tool_invocation_content(response):
    tool_invocation_content: str = ""
    for chunk in response:
        delta = chunk.event.delta
        if delta.type == "tool_call" and delta.parse_status == "succeeded":
            call = delta.content
            tool_invocation_content += f"[{call.tool_name}, {call.arguments}]"
    return tool_invocation_content


def test_text_chat_completion_with_tool_calling_and_streaming(
    llama_stack_client, text_model_id, get_weather_tool_definition, provider_tool_format
):
    response = llama_stack_client.inference.chat_completion(
        model_id=text_model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's the weather like in San Francisco?"},
        ],
        tools=[get_weather_tool_definition],
        tool_choice="auto",
        tool_prompt_format=provider_tool_format,
        stream=True,
    )
    tool_invocation_content = extract_tool_invocation_content(response)
    assert tool_invocation_content == "[get_weather, {'location': 'San Francisco, CA'}]"


def test_text_chat_completion_structured_output(
    llama_stack_client, text_model_id, inference_provider_type
):
    class AnswerFormat(BaseModel):
        first_name: str
        last_name: str
        year_of_birth: int
        num_seasons_in_nba: int

    response = llama_stack_client.inference.chat_completion(
        model_id=text_model_id,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Michael Jordan was born in 1963. He played basketball for the Chicago Bulls for 15 seasons.",
            },
            {
                "role": "user",
                "content": "Please give me information about Michael Jordan.",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": AnswerFormat.model_json_schema(),
        },
        stream=False,
    )
    answer = AnswerFormat.model_validate_json(response.completion_message.content)
    assert answer.first_name == "Michael"
    assert answer.last_name == "Jordan"
    assert answer.year_of_birth == 1963
    assert answer.num_seasons_in_nba == 15


def test_image_chat_completion_non_streaming(llama_stack_client, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": {
                    # TODO: Replace with Github based URI to resources/sample1.jpg
                    "uri": "https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = llama_stack_client.inference.chat_completion(
        model_id=vision_model_id,
        messages=[message],
        stream=False,
    )
    message_content = response.completion_message.content.lower().strip()
    assert len(message_content) > 0
    assert any(expected in message_content for expected in {"dog", "puppy", "pup"})


def test_image_chat_completion_streaming(llama_stack_client, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": {
                    # TODO: Replace with Github based URI to resources/sample1.jpg
                    "uri": "https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = llama_stack_client.inference.chat_completion(
        model_id=vision_model_id,
        messages=[message],
        stream=True,
    )
    streamed_content = ""
    for chunk in response:
        streamed_content += chunk.event.delta.text.lower()
    assert len(streamed_content) > 0
    assert any(expected in streamed_content for expected in {"dog", "puppy", "pup"})
