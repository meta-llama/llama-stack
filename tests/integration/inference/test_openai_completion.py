# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import base64
import os
import tempfile

import pytest
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

from ..test_cases.test_case import TestCase


def provider_from_model(client_with_models, model_id):
    models = {m.identifier: m for m in client_with_models.models.list()}
    models.update({m.provider_resource_id: m for m in client_with_models.models.list()})
    provider_id = models[model_id].provider_id
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    return providers[provider_id]


def skip_if_model_doesnt_support_openai_completion(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type in (
        "inline::meta-reference",
        "inline::sentence-transformers",
        "inline::vllm",
        "remote::bedrock",
        "remote::cerebras",
        "remote::databricks",
        # Technically Nvidia does support OpenAI completions, but none of their hosted models
        # support both completions and chat completions endpoint and all the Llama models are
        # just chat completions
        "remote::nvidia",
        "remote::runpod",
        "remote::sambanova",
        "remote::tgi",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support OpenAI completions.")


def skip_if_model_doesnt_support_suffix(client_with_models, model_id):
    # To test `fim` ( fill in the middle ) completion, we need to use a model that supports suffix.
    # Use this to specifically test this API functionality.

    # pytest -sv --stack-config="inference=starter" \
    # tests/integration/inference/test_openai_completion.py \
    # --text-model qwen2.5-coder:1.5b \
    # -k test_openai_completion_non_streaming_suffix

    if model_id != "qwen2.5-coder:1.5b":
        pytest.skip(f"Suffix is not supported for the model: {model_id}.")

    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type != "remote::ollama":
        pytest.skip(f"Provider {provider.provider_type} doesn't support suffix.")


def skip_if_model_doesnt_support_openai_chat_completion(client_with_models, model_id):
    if isinstance(client_with_models, LlamaStackAsLibraryClient):
        pytest.skip("OpenAI chat completions are not supported when testing with library client yet.")

    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type in (
        "inline::meta-reference",
        "inline::sentence-transformers",
        "inline::vllm",
        "remote::bedrock",
        "remote::cerebras",
        "remote::databricks",
        "remote::runpod",
        "remote::tgi",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support OpenAI chat completions.")


def skip_if_provider_isnt_vllm(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type != "remote::vllm":
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support vllm extra_body parameters.")


def skip_if_provider_isnt_openai(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type != "remote::openai":
        pytest.skip(
            f"Model {model_id} hosted by {provider.provider_type} doesn't support chat completion calls with base64 encoded files."
        )


@pytest.fixture
def openai_client(client_with_models):
    base_url = f"{client_with_models.base_url}/v1/openai/v1"
    return OpenAI(base_url=base_url, api_key="bar")


@pytest.fixture(params=["openai_client", "llama_stack_client"])
def compat_client(request):
    return request.getfixturevalue(request.param)


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:sanity",
    ],
)
def test_openai_completion_non_streaming(llama_stack_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)

    # ollama needs more verbose prompting for some reason here...
    prompt = "Respond to this question and explain your answer. " + tc["content"]
    response = llama_stack_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=False,
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert len(choice.text) > 10


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:suffix",
    ],
)
def test_openai_completion_non_streaming_suffix(llama_stack_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    skip_if_model_doesnt_support_suffix(client_with_models, text_model_id)
    tc = TestCase(test_case)

    # ollama needs more verbose prompting for some reason here...
    response = llama_stack_client.completions.create(
        model=text_model_id,
        prompt=tc["content"],
        stream=False,
        suffix=tc["suffix"],
        max_tokens=10,
    )

    assert len(response.choices) > 0
    choice = response.choices[0]
    assert len(choice.text) > 5
    assert "france" in choice.text.lower()


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:sanity",
    ],
)
def test_openai_completion_streaming(llama_stack_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)

    # ollama needs more verbose prompting for some reason here...
    prompt = "Respond to this question and explain your answer. " + tc["content"]
    response = llama_stack_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=True,
        max_tokens=50,
    )
    streamed_content = [chunk.choices[0].text or "" for chunk in response]
    content_str = "".join(streamed_content).lower().strip()
    assert len(content_str) > 10


@pytest.mark.parametrize(
    "prompt_logprobs",
    [
        1,
        0,
    ],
)
def test_openai_completion_prompt_logprobs(llama_stack_client, client_with_models, text_model_id, prompt_logprobs):
    skip_if_provider_isnt_vllm(client_with_models, text_model_id)

    prompt = "Hello, world!"
    response = llama_stack_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=False,
        prompt_logprobs=prompt_logprobs,
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert len(choice.prompt_logprobs) > 0


def test_openai_completion_guided_choice(llama_stack_client, client_with_models, text_model_id):
    skip_if_provider_isnt_vllm(client_with_models, text_model_id)

    prompt = "I am feeling really sad today."
    response = llama_stack_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=False,
        guided_choice=["joy", "sadness"],
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert choice.text in ["joy", "sadness"]


# Run the chat-completion tests with both the OpenAI client and the LlamaStack client


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:non_streaming_01",
        "inference:chat_completion:non_streaming_02",
    ],
)
def test_openai_chat_completion_non_streaming(compat_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = compat_client.chat.completions.create(
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ],
        stream=False,
    )
    message_content = response.choices[0].message.content.lower().strip()
    assert len(message_content) > 0
    assert expected.lower() in message_content


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:streaming_01",
        "inference:chat_completion:streaming_02",
    ],
)
def test_openai_chat_completion_streaming(compat_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = compat_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": question}],
        stream=True,
        timeout=120,  # Increase timeout to 2 minutes for large conversation history
    )
    streamed_content = []
    for chunk in response:
        if chunk.choices[0].delta.content:
            streamed_content.append(chunk.choices[0].delta.content.lower().strip())
    assert len(streamed_content) > 0
    assert expected.lower() in "".join(streamed_content)


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:streaming_01",
        "inference:chat_completion:streaming_02",
    ],
)
def test_openai_chat_completion_streaming_with_n(compat_client, client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)

    provider = provider_from_model(client_with_models, text_model_id)
    if provider.provider_type == "remote::ollama":
        pytest.skip(f"Model {text_model_id} hosted by {provider.provider_type} doesn't support n > 1.")

    tc = TestCase(test_case)
    question = tc["question"]
    expected = tc["expected"]

    response = compat_client.chat.completions.create(
        model=text_model_id,
        messages=[{"role": "user", "content": question}],
        stream=True,
        timeout=120,  # Increase timeout to 2 minutes for large conversation history,
        n=2,
    )
    streamed_content = {}
    for chunk in response:
        for choice in chunk.choices:
            if choice.delta.content:
                streamed_content[choice.index] = (
                    streamed_content.get(choice.index, "") + choice.delta.content.lower().strip()
                )
    assert len(streamed_content) == 2
    for i, content in streamed_content.items():
        assert expected.lower() in content, f"Choice {i}: Expected {expected.lower()} in {content}"


@pytest.mark.parametrize(
    "stream",
    [
        True,
        False,
    ],
)
def test_inference_store(compat_client, client_with_models, text_model_id, stream):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    client = compat_client
    # make a chat completion
    message = "Hello, world!"
    response = client.chat.completions.create(
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        stream=stream,
    )
    if stream:
        # accumulate the streamed content
        content = ""
        response_id = None
        for chunk in response:
            if response_id is None:
                response_id = chunk.id
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
    else:
        response_id = response.id
        content = response.choices[0].message.content

    responses = client.chat.completions.list()
    assert response_id in [r.id for r in responses.data]

    retrieved_response = client.chat.completions.retrieve(response_id)
    assert retrieved_response.id == response_id
    assert retrieved_response.choices[0].message.content == content, retrieved_response

    input_content = (
        getattr(retrieved_response.input_messages[0], "content", None)
        or retrieved_response.input_messages[0]["content"]
    )
    assert input_content == message, retrieved_response


@pytest.mark.parametrize(
    "stream",
    [
        True,
        False,
    ],
)
def test_inference_store_tool_calls(compat_client, client_with_models, text_model_id, stream):
    skip_if_model_doesnt_support_openai_chat_completion(client_with_models, text_model_id)
    client = compat_client
    # make a chat completion
    message = "What's the weather in Tokyo? Use the get_weather function to get the weather."
    response = client.chat.completions.create(
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        stream=stream,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather in a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "The city to get the weather for"},
                        },
                    },
                },
            }
        ],
    )
    if stream:
        # accumulate the streamed content
        content = ""
        response_id = None
        for chunk in response:
            if response_id is None:
                response_id = chunk.id
            if delta := chunk.choices[0].delta:
                if delta.content:
                    content += delta.content
    else:
        response_id = response.id
        content = response.choices[0].message.content

    responses = client.chat.completions.list()
    assert response_id in [r.id for r in responses.data]

    retrieved_response = client.chat.completions.retrieve(response_id)
    assert retrieved_response.id == response_id
    input_content = (
        getattr(retrieved_response.input_messages[0], "content", None)
        or retrieved_response.input_messages[0]["content"]
    )
    assert input_content == message, retrieved_response
    tool_calls = retrieved_response.choices[0].message.tool_calls
    # sometimes model doesn't output tool calls, but we still want to test that the tool was called
    if tool_calls:
        # because we test with small models, just check that we retrieved
        # a tool call with a name and arguments string, but ignore contents
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name
        assert tool_calls[0].function.arguments
    else:
        # failed tool call parses show up as a message with content, so ensure
        # that the retrieve response content matches the original request
        assert retrieved_response.choices[0].message.content == content


def test_openai_chat_completion_non_streaming_with_file(openai_client, client_with_models, text_model_id):
    skip_if_provider_isnt_openai(client_with_models, text_model_id)

    # Generate temporary PDF with "Hello World" text
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        c = canvas.Canvas(temp_pdf.name, pagesize=letter)
        c.drawString(100, 750, "Hello World")
        c.save()

        # Read the PDF and sencode to base64
        with open(temp_pdf.name, "rb") as pdf_file:
            pdf_base64 = base64.b64encode(pdf_file.read()).decode("utf-8")

        # Clean up temporary file
        os.unlink(temp_pdf.name)

    response = openai_client.chat.completions.create(
        model=text_model_id,
        messages=[
            {
                "role": "user",
                "content": "Describe what you see in this PDF file.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": "my-temp-hello-world-pdf",
                            "file_data": f"data:application/pdf;base64,{pdf_base64}",
                        },
                    }
                ],
            },
        ],
        stream=False,
    )
    message_content = response.choices[0].message.content.lower().strip()
    assert "hello world" in message_content
