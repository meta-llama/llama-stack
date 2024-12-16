# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from llama_stack_client.lib.inference.event_logger import EventLogger


def test_text_chat_completion(llama_stack_client):
    # non-streaming
    available_models = [model.identifier for model in llama_stack_client.models.list()]
    assert len(available_models) > 0
    model_id = available_models[0]
    response = llama_stack_client.inference.chat_completion(
        model_id=model_id,
        messages=[
            {
                "role": "user",
                "content": "Hello, world!",
            }
        ],
        stream=False,
    )
    assert len(response.completion_message.content) > 0

    # streaming
    response = llama_stack_client.inference.chat_completion(
        model_id=model_id,
        messages=[{"role": "user", "content": "Hello, world!"}],
        stream=True,
    )
    logs = [str(log.content) for log in EventLogger().log(response) if log is not None]
    assert len(logs) > 0
    assert "Assistant> " in logs[0]


def test_image_chat_completion(llama_stack_client):
    available_models = [
        model.identifier
        for model in llama_stack_client.models.list()
        if "vision" in model.identifier.lower()
    ]
    if len(available_models) == 0:
        pytest.skip("No vision models available")

    model_id = available_models[0]
    # non-streaming
    message = {
        "role": "user",
        "content": [
            {
                "image": {
                    "uri": "https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                }
            },
            "Describe what is in this image.",
        ],
    }
    response = llama_stack_client.inference.chat_completion(
        model_id=model_id,
        messages=[message],
        stream=False,
    )
    assert len(response.completion_message.content) > 0
    assert (
        "dog" in response.completion_message.content.lower()
        or "puppy" in response.completion_message.content.lower()
    )
