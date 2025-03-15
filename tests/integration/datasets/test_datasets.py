# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import pytest

# How to run this test:
#
# LLAMA_STACK_CONFIG="template-name" pytest -v tests/integration/datasets


@pytest.mark.parametrize(
    "purpose, source, provider_id, limit",
    [
        (
            "eval/messages-answer",
            {
                "type": "uri",
                "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
            },
            "huggingface",
            10,
        ),
        (
            "eval/messages-answer",
            {
                "type": "rows",
                "rows": [
                    {
                        "messages": [{"role": "user", "content": "Hello, world!"}],
                        "answer": "Hello, world!",
                    },
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "What is the capital of France?",
                            }
                        ],
                        "answer": "Paris",
                    },
                ],
            },
            "localfs",
            2,
        ),
    ],
)
def test_register_and_iterrows(llama_stack_client, purpose, source, provider_id, limit):
    dataset = llama_stack_client.datasets.register(
        purpose=purpose,
        source=source,
    )
    assert dataset.identifier is not None
    assert dataset.provider_id == provider_id
    iterrow_response = llama_stack_client.datasets.iterrows(
        dataset.identifier, limit=limit
    )
    assert len(iterrow_response.data) == limit

    dataset_list = llama_stack_client.datasets.list()
    assert dataset.identifier in [d.identifier for d in dataset_list]

    llama_stack_client.datasets.unregister(dataset.identifier)
    dataset_list = llama_stack_client.datasets.list()
    assert dataset.identifier not in [d.identifier for d in dataset_list]
