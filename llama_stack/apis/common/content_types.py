# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

from llama_models.llama3.api.datatypes import ToolCall

from llama_models.schema_utils import json_schema_type, register_schema
from pydantic import BaseModel, Field, field_serializer, model_validator


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

    @field_serializer("data")
    def serialize_data(self, data: Optional[bytes], _info):
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")


@json_schema_type
class ImageContentItem(BaseModel):
    type: Literal["image"] = "image"
    image: _URLOrData


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


@json_schema_type
class TextDelta(BaseModel):
    type: Literal["text"] = "text"
    text: str


@json_schema_type
class ImageDelta(BaseModel):
    type: Literal["image"] = "image"
    image: bytes


@json_schema_type
class ToolCallParseStatus(Enum):
    started = "started"
    in_progress = "in_progress"
    failed = "failed"
    succeeded = "succeeded"


@json_schema_type
class ToolCallDelta(BaseModel):
    type: Literal["tool_call"] = "tool_call"

    # you either send an in-progress tool call so the client can stream a long
    # code generation or you send the final parsed tool call at the end of the
    # stream
    tool_call: Union[str, ToolCall]
    parse_status: ToolCallParseStatus


# streaming completions send a stream of ContentDeltas
ContentDelta = register_schema(
    Annotated[
        Union[TextDelta, ImageDelta, ToolCallDelta],
        Field(discriminator="type"),
    ],
    name="ContentDelta",
)
