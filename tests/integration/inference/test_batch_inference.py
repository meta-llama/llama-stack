# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

from ..test_cases.test_case import TestCase


def skip_if_provider_doesnt_support_batch_inference(client_with_models, model_id):
    models = {m.identifier: m for m in client_with_models.models.list()}
    models.update({m.provider_resource_id: m for m in client_with_models.models.list()})
    provider_id = models[model_id].provider_id
    providers = {p.provider_id: p for p in client_with_models.providers.list()}
    provider = providers[provider_id]
    if provider.provider_type not in ("inline::meta-reference",):
        pytest.skip(f"Model {model_id} hosted by {provider.provider_type} doesn't support batch inference")


@pytest.mark.parametrize(
    "test_case",
    [
        "inference:completion:batch_completion",
    ],
)
def test_batch_completion_non_streaming(client_with_models, text_model_id, test_case):
    skip_if_provider_doesnt_support_batch_inference(client_with_models, text_model_id)
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
    skip_if_provider_doesnt_support_batch_inference(client_with_models, text_model_id)
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
