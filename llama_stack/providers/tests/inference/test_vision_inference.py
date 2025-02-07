# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
from pathlib import Path

import pytest

from llama_stack.apis.common.content_types import URL, ImageContentItem, TextContentItem
from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    SamplingParams,
    UserMessage,
)

from .utils import group_chunks

THIS_DIR = Path(__file__).parent

with open(THIS_DIR / "pasta.jpeg", "rb") as f:
    PASTA_IMAGE = base64.b64encode(f.read()).decode("utf-8")


class TestVisionModelInference:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "image, expected_strings",
        [
            (
                ImageContentItem(image=dict(data=PASTA_IMAGE)),
                ["spaghetti"],
            ),
            (
                ImageContentItem(
                    image=dict(
                        url=URL(
                            uri="https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                        )
                    )
                ),
                ["puppy"],
            ),
        ],
    )
    async def test_vision_chat_completion_non_streaming(
        self, inference_model, inference_stack, image, expected_strings
    ):
        inference_impl, _ = inference_stack
        response = await inference_impl.chat_completion(
            model_id=inference_model,
            messages=[
                UserMessage(content="You are a helpful assistant."),
                UserMessage(
                    content=[
                        image,
                        TextContentItem(text="Describe this image in two sentences."),
                    ]
                ),
            ],
            stream=False,
            sampling_params=SamplingParams(max_tokens=100),
        )

        assert isinstance(response, ChatCompletionResponse)
        assert response.completion_message.role == "assistant"
        assert isinstance(response.completion_message.content, str)
        for expected_string in expected_strings:
            assert expected_string in response.completion_message.content

    @pytest.mark.asyncio
    async def test_vision_chat_completion_streaming(self, inference_model, inference_stack):
        inference_impl, _ = inference_stack

        images = [
            ImageContentItem(
                image=dict(
                    url=URL(
                        uri="https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                    )
                )
            ),
        ]
        expected_strings_to_check = [
            ["puppy"],
        ]
        for image, expected_strings in zip(images, expected_strings_to_check):
            response = [
                r
                async for r in await inference_impl.chat_completion(
                    model_id=inference_model,
                    messages=[
                        UserMessage(content="You are a helpful assistant."),
                        UserMessage(
                            content=[
                                image,
                                TextContentItem(text="Describe this image in two sentences."),
                            ]
                        ),
                    ],
                    stream=True,
                    sampling_params=SamplingParams(max_tokens=100),
                )
            ]

            assert len(response) > 0
            assert all(isinstance(chunk, ChatCompletionResponseStreamChunk) for chunk in response)
            grouped = group_chunks(response)
            assert len(grouped[ChatCompletionResponseEventType.start]) == 1
            assert len(grouped[ChatCompletionResponseEventType.progress]) > 0
            assert len(grouped[ChatCompletionResponseEventType.complete]) == 1

            content = "".join(chunk.event.delta.text for chunk in grouped[ChatCompletionResponseEventType.progress])
            for expected_string in expected_strings:
                assert expected_string in content
