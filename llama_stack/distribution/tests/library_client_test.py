# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage


def main(config_path: str):
    client = LlamaStackAsLibraryClient(config_path)
    client.initialize()

    models = client.models.list()
    print("\nModels:")
    for model in models:
        print(model)

    if not models:
        print("No models found, skipping chat completion test")
        return

    model_id = models[0].identifier
    response = client.inference.chat_completion(
        messages=[UserMessage(content="What is the capital of France?", role="user")],
        model_id=model_id,
        stream=False,
    )
    print("\nChat completion response (non-stream):")
    print(response)

    response = client.inference.chat_completion(
        messages=[UserMessage(content="What is the capital of France?", role="user")],
        model_id=model_id,
        stream=True,
    )

    print("\nChat completion response (stream):")
    for log in EventLogger().log(response):
        log.print()

    response = client.memory_banks.register(
        memory_bank_id="memory_bank_id",
        params={
            "chunk_size_in_tokens": 0,
            "embedding_model": "embedding_model",
            "memory_bank_type": "vector",
        },
    )
    print("\nRegister memory bank response:")
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()
    main(args.config_path)
