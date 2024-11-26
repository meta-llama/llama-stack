# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from llama_stack_client import LlamaStackClient

from pydantic import BaseModel

client = LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")


class CompletionMessage(BaseModel):
    recipe_name: str
    preamble: str
    ingredients: list[str]
    steps: list[str]


response = client.inference.chat_completion(
    model_id=os.environ["INFERENCE_MODEL"],
    messages=[
        {
            "role": "system",
            "content": "You are a chef, passionate about educating the world about delicious home cooked meals.",
        },
        {
            "role": "user",
            "content": "Give me a recipe for spaghetti bolognaise. Start with the recipe name, a preamble describing your childhood stories about spaghetti bolognaise, an ingredients list, and then the recipe steps.",
        },
    ],
    response_format={
        "type": "json_schema",
        "json_schema": CompletionMessage.model_json_schema(),
    },
    sampling_params={"max_tokens": 8000},
)
print(response.completion_message.content)
