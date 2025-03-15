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
    "purpose, source, provider_id",
    [
        (
            "eval/messages-answer",
            {
                "type": "uri",
                "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
            },
            "huggingface",
        ),
    ],
)
def test_register_dataset(llama_stack_client, purpose, source, provider_id):
    dataset = llama_stack_client.datasets.register(
        purpose=purpose,
        source=source,
    )
    assert dataset.identifier is not None
    assert dataset.provider_id == provider_id
    iterrow_response = llama_stack_client.datasets.iterrows(dataset.identifier, limit=10)
    assert len(iterrow_response.data) == 10
    assert iterrow_response.next_index is not None
