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
        "remote::nvidia",
        "remote::runpod",
        "remote::sambanova",
        "remote::tgi",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support OpenAI completions.")


def skip_if_provider_isnt_vllm(client_with_models, model_id):
    provider = provider_from_model(client_with_models, model_id)
    if provider.provider_type != "remote::vllm":
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support vllm extra_body parameters.")


@pytest.fixture
def openai_client(client_with_models, text_model_id):
    skip_if_model_doesnt_support_openai_completion(client_with_models, text_model_id)
    base_url = f"{client_with_models.base_url}/v1/openai/v1"
    return OpenAI(base_url=base_url, api_key="bar")


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:sanity",
    ],
)
def test_openai_completion_non_streaming(openai_client, text_model_id, test_case):
    tc = TestCase(test_case)

    # ollama needs more verbose prompting for some reason here...
    prompt = "Respond to this question and explain your answer. " + tc["content"]
    response = openai_client.completions.create(
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
def test_openai_completion_streaming(openai_client, text_model_id, test_case):
    tc = TestCase(test_case)

    # ollama needs more verbose prompting for some reason here...
    prompt = "Respond to this question and explain your answer. " + tc["content"]
    response = openai_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=True,
        max_tokens=50,
    )
    streamed_content = [chunk.choices[0].text for chunk in response]
    content_str = "".join(streamed_content).lower().strip()
    assert len(content_str) > 10


def test_openai_completion_prompt_logprobs(openai_client, client_with_models, text_model_id):
    skip_if_provider_isnt_vllm(client_with_models, text_model_id)

    prompt = "Hello, world!"
    response = openai_client.completions.create(
        model=text_model_id,
        prompt=prompt,
        stream=False,
        extra_body={
            "prompt_logprobs": 1,
        },
    )
    assert len(response.choices) > 0
    choice = response.choices[0]
    assert len(choice.prompt_logprobs) > 0


def test_openai_completion_guided_choice(openai_client, client_with_models, text_model_id):
    skip_if_provider_isnt_vllm(client_with_models, text_model_id)

    prompt = "I am feeling really sad today."
    response = openai_client.completions.create(
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
