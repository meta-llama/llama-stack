# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import pathlib

import pytest


@pytest.fixture(scope="session")
def inference_provider_type(llama_stack_client):
    providers = llama_stack_client.providers.list()
    inference_providers = [p for p in providers if p.api == "inference"]
    assert len(inference_providers) > 0, "No inference providers found"
    return inference_providers[0].provider_type


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


def test_image_chat_completion_non_streaming(llama_stack_client, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": {
                    "url": {
                        # TODO: Replace with Github based URI to resources/sample1.jpg
                        "uri": "https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                    },
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = llama_stack_client.inference.chat_completion(
        model_id=vision_model_id,
        messages=[message],
        stream=False,
    )
    message_content = response.completion_message.content.lower().strip()
    assert len(message_content) > 0
    assert any(expected in message_content for expected in {"dog", "puppy", "pup"})


def test_image_chat_completion_streaming(llama_stack_client, vision_model_id):
    message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": {
                    "url": {
                        # TODO: Replace with Github based URI to resources/sample1.jpg
                        "uri": "https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                    },
                },
            },
            {
                "type": "text",
                "text": "Describe what is in this image.",
            },
        ],
    }
    response = llama_stack_client.inference.chat_completion(
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
def test_image_chat_completion_base64(llama_stack_client, vision_model_id, base64_image_data, base64_image_url, type_):
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
    response = llama_stack_client.inference.chat_completion(
        model_id=vision_model_id,
        messages=[message],
        stream=False,
    )
    message_content = response.completion_message.content.lower().strip()
    assert len(message_content) > 0
