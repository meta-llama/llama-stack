# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, List, Literal, Union

from llama_models.schema_utils import json_schema_type, register_schema

from pydantic import BaseModel, Field


@json_schema_type(
    schema={"type": "string", "format": "uri", "pattern": "^(https?://|file://|data:)"}
)
class URL(BaseModel):
    uri: str

    def __str__(self) -> str:
        return self.uri


@json_schema_type
class ImageContentItem(BaseModel):
    type: Literal["image"] = "image"
    data: Union[bytes, URL]


@json_schema_type
class TextContentItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


# other modalities can be added here
InterleavedContentItem = register_schema(
    Annotated[
        Union[ImageContentItem, TextContentItem],
        Field(discriminator="type"),
    ],
    name="InterleavedContentItem",
)

# accept a single "str" as a special case since it is common
InterleavedContent = register_schema(
    Union[str, InterleavedContentItem, List[InterleavedContentItem]],
    name="InterleavedContent",
)
