# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from llama_stack.models.llama.sku_list import resolve_model

from ..test_cases.test_case import TestCase

PROVIDER_LOGPROBS_TOP_K = {"remote::together", "remote::fireworks", "remote::vllm"}


def skip_if_model_doesnt_support_completion(client_with_models, model_id):
    models = {m.identifier: m for m in client_with_models.models.list()}
    models.update({m.provider_resource_id: m for m in client_with_models.models.list()})
    provider_id = models[model_id].provider_id
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    provider = providers[provider_id]
    if provider.provider_type in (
        "remote::openai",
        "remote::anthropic",
        "remote::gemini",
        "remote::groq",
        "remote::llama-openai-compat",
    ):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support completion")


def get_llama_model(client_with_models, model_id):
    models = {}
    for m in client_with_models.models.list():
        models[m.identifier] = m
        models[m.provider_resource_id] = m

    assert model_id in models, f"Model {model_id} not found"

    model = models[model_id]
    ids = (model.identifier, model.provider_resource_id)
    for mid in ids:
        if resolve_model(mid):
            return mid

    return model.metadata.get("llama_model", None)


def get_llama_tokenizer():
    from llama_models.llama3.api.chat_format import ChatFormat
    from llama_models.llama3.api.tokenizer import Tokenizer

    tokenizer = Tokenizer.get_instance()
    formatter = ChatFormat(tokenizer)
    return tokenizer, formatter


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:batch_completion",
    ],
)
def test_batch_completion_non_streaming(client_with_models, text_model_id, test_case):
    skip_if_model_doesnt_support_completion(client_with_models, text_model_id)
    tc = TestCase(test_case)

    content_batch = tc["contents"]
    response = client_with_models.inference.batch_completion(
        content_batch=content_batch,
        model_id=text_model_id,
        sampling_params={
            "max_tokens": 50,
        },
    )
    assert len(response.batch) == len(content_batch)
    for i, r in enumerate(response.batch):
        print(f"response {i}: {r.content}")
        assert len(r.content) > 10


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:chat_completion:batch_completion",
    ],
)
def test_batch_chat_completion_non_streaming(client_with_models, text_model_id, test_case):
    tc = TestCase(test_case)
    qa_pairs = tc["qa_pairs"]

    message_batch = [
        [
            {
                "role": "user",
                "content": qa["question"],
            }
        ]
        for qa in qa_pairs
    ]

    response = client_with_models.inference.batch_chat_completion(
        messages_batch=message_batch,
        model_id=text_model_id,
    )
    assert len(response.batch) == len(qa_pairs)
    for i, r in enumerate(response.batch):
        print(f"response {i}: {r.completion_message.content}")
        assert len(r.completion_message.content) > 0
        assert qa_pairs[i]["answer"].lower() in r.completion_message.content.lower()
