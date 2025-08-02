# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type, register_schema


@json_schema_type
class OpenAIResponseInputMessageContentText(BaseModel):
    """Text content for input messages in OpenAI response format.

    :param text: The text content of the input message
    :param type: Content type identifier, always "input_text"
    """

    text: str
    type: Literal["input_text"] = "input_text"


@json_schema_type
class OpenAIResponseInputMessageContentImage(BaseModel):
    """Image content for input messages in OpenAI response format.

    :param detail: Level of detail for image processing, can be "low", "high", or "auto"
    :param type: Content type identifier, always "input_image"
    :param image_url: (Optional) URL of the image content
    """

    detail: Literal["low"] | Literal["high"] | Literal["auto"] = "auto"
    type: Literal["input_image"] = "input_image"
    # TODO: handle file_id
    image_url: str | None = None


# TODO: handle file content types
OpenAIResponseInputMessageContent = Annotated[
    OpenAIResponseInputMessageContentText | OpenAIResponseInputMessageContentImage,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputMessageContent, name="OpenAIResponseInputMessageContent")
