# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest
from openai import OpenAI

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

from ..test_cases.test_case import TestCase


def skip_if_model_doesnt_support_openai_completion(client_with_models, model_id):
    if isinstance(client_with_models, LlamaStackAsLibraryClient):
        pytest.skip("OpenAI completions are not supported when testing with library client yet.")

    models = {m.identifier: m for m in client_with_models.models.list()}
    models.update({m.provider_resource_id: m for m in client_with_models.models.list()})
    provider_id = models[model_id].provider_id
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    provider = providers[provider_id]
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

    response = openai_client.completions.create(
        model=text_model_id,
        prompt=tc["content"],
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

    response = openai_client.completions.create(
        model=text_model_id,
        prompt=tc["content"],
        stream=True,
        max_tokens=50,
    )
    streamed_content = [chunk.choices[0].text for chunk in response]
    content_str = "".join(streamed_content).lower().strip()
    assert len(content_str) > 10
