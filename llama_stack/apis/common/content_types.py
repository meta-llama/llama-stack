# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, List, Literal, Optional, Union

from llama_models.schema_utils import json_schema_type, register_schema

from pydantic import BaseModel, Field, model_validator


@json_schema_type
class URL(BaseModel):
    uri: str


class _URLOrData(BaseModel):
    url: Optional[URL] = None
    data: Optional[bytes] = None

    @model_validator(mode="before")
    @classmethod
    def validator(cls, values):
        if isinstance(values, dict):
            return values
        return {"url": values}


@json_schema_type
class ImageContentItem(_URLOrData):
    type: Literal["image"] = "image"


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
