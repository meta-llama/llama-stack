# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest
from openai import OpenAI

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

from ..test_cases.test_case import TestCase


def provider_from_model(client_with_models, model_id):
    models = {m.identifier: m for m in client_with_models.models.list()}
    models.update({m.provider_resource_id: m for m in client_with_models.models.list()})
    provider_id = models[model_id].provider_id
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    return providers[provider_id]


def skip_if_model_doesnt_support_openai_completion(client_with_models, model_id):
    if isinstance(client_with_models, LlamaStackAsLibraryClient):
        pytest.skip("OpenAI completions are not supported when testing with library client yet.")

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
        "remote::sambanova",
        "remote::tgi",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support OpenAI chat completions.")


def skip_if_provider_isnt_vllm(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type != "remote::vllm":
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support vllm extra_body parameters.")


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
        extra_body={
            "prompt_logprobs": prompt_logprobs,
        },
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
        extra_body={
            "guided_choice": ["joy", "sadness"],
        },
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
