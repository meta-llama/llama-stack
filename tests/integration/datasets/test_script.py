# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client import LlamaStackClient
from rich.pretty import pprint


def test_register_dataset():
    client = LlamaStackClient(base_url="http://localhost:8321")
    # dataset = client.datasets.register(
    #     purpose="eval/messages-answer",
    #     source={
    #         "type": "uri",
    #         "uri": "huggingface://datasets/llamastack/simpleqa?split=train",
    #     },
    # )
    dataset = client.datasets.register(
        purpose="eval/messages-answer",
        source={
            "type": "rows",
            "rows": [
                {
                    "messages": [{"role": "user", "content": "Hello, world!"}],
                    "answer": "Hello, world!",
                },
                {
                    "messages": [{"role": "user", "content": "What is the capital of France?"}],
                    "answer": "Paris",
                },
            ],
        },
    )
    dataset_id = dataset.identifier
    pprint(dataset)
    rows = client.datasets.iterrows(dataset_id=dataset_id, limit=10)
    pprint(rows)


if __name__ == "__main__":
    test_register_dataset()
