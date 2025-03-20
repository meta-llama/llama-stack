# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import pathlib

import pytest


@pytest.fixture
def image_path():
    return pathlib.Path(__file__).parent / "dog.png"


@pytest.fixture
def base64_image_data(image_path):
    # Convert the image to base64
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


@pytest.fixture
def base64_image_url(base64_image_data, image_path):
    # suffix includes the ., so we remove it
    return f"data:image/{image_path.suffix[1:]};base64,{base64_image_data}"


@pytest.mark.xfail(reason="This test is failing because the image is not being downloaded correctly.")
def test_image_chat_completion_non_streaming(client_with_models, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": {
                    "url": {
                        "uri": "https://raw.githubusercontent.com/meta-llama/llama-stack/main/tests/integration/inference/dog.png"
                    },
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = client_with_models.inference.chat_completion(
        model_id=vision_model_id,
        messages=[message],
        stream=False,
    )
    message_content = response.completion_message.content.lower().strip()
    assert len(message_content) > 0
    assert any(expected in message_content for expected in {"dog", "puppy", "pup"})


@pytest.mark.xfail(reason="This test is failing because the image is not being downloaded correctly.")
def test_image_chat_completion_streaming(client_with_models, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": {
                    "url": {
                        "uri": "https://raw.githubusercontent.com/meta-llama/llama-stack/main/tests/integration/inference/dog.png"
                    },
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = client_with_models.inference.chat_completion(
        model_id=vision_model_id,
        messages=[message],
        stream=True,
    )
    streamed_content = ""
    for chunk in response:
        streamed_content += chunk.event.delta.text.lower()
    assert len(streamed_content) > 0
    assert any(expected in streamed_content for expected in {"dog", "puppy", "pup"})


@pytest.mark.parametrize("type_", ["url", "data"])
def test_image_chat_completion_base64(client_with_models, vision_model_id, base64_image_data, base64_image_url, type_):
    image_spec = {
        "url": {
            "type": "image",
            "image": {
                "url": {
                    "uri": base64_image_url,
                },
            },
        },
        "data": {
            "type": "image",
            "image": {
                "data": base64_image_data,
            },
        },
    }[type_]

    message = {
        "role": "user",
        "content": [
            image_spec,
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = client_with_models.inference.chat_completion(
        model_id=vision_model_id,
        messages=[message],
        stream=False,
    )
    message_content = response.completion_message.content.lower().strip()
    assert len(message_content) > 0
