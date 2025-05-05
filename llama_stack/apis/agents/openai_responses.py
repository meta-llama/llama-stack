# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from llama_stack.schema_utils import json_schema_type, register_schema


@json_schema_type
class OpenAIResponseError(BaseModel):
    code: str
    message: str


@json_schema_type
class OpenAIResponseOutputMessageContentOutputText(BaseModel):
    text: str
    type: Literal["output_text"] = "output_text"


OpenAIResponseOutputMessageContent = Annotated[
    OpenAIResponseOutputMessageContentOutputText,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutputMessageContent, name="OpenAIResponseOutputMessageContent")


@json_schema_type
class OpenAIResponseOutputMessage(BaseModel):
    id: str
    content: list[OpenAIResponseOutputMessageContent]
    role: Literal["assistant"] = "assistant"
    status: str
    type: Literal["message"] = "message"


@json_schema_type
class OpenAIResponseOutputMessageWebSearchToolCall(BaseModel):
    id: str
    status: str
    type: Literal["web_search_call"] = "web_search_call"


OpenAIResponseOutput = Annotated[
    OpenAIResponseOutputMessage | OpenAIResponseOutputMessageWebSearchToolCall,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutput, name="OpenAIResponseOutput")


@json_schema_type
class OpenAIResponseObject(BaseModel):
    created_at: int
    error: OpenAIResponseError | None = None
    id: str
    model: str
    object: Literal["response"] = "response"
    output: list[OpenAIResponseOutput]
    parallel_tool_calls: bool = False
    previous_response_id: str | None = None
    status: str
    temperature: float | None = None
    top_p: float | None = None
    truncation: str | None = None
    user: str | None = None


@json_schema_type
class OpenAIResponseObjectStreamResponseCreated(BaseModel):
    response: OpenAIResponseObject
    type: Literal["response.created"] = "response.created"


@json_schema_type
class OpenAIResponseObjectStreamResponseCompleted(BaseModel):
    response: OpenAIResponseObject
    type: Literal["response.completed"] = "response.completed"


OpenAIResponseObjectStream = Annotated[
    OpenAIResponseObjectStreamResponseCreated | OpenAIResponseObjectStreamResponseCompleted,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseObjectStream, name="OpenAIResponseObjectStream")


@json_schema_type
class OpenAIResponseInputMessageContentText(BaseModel):
    text: str
    type: Literal["input_text"] = "input_text"


@json_schema_type
class OpenAIResponseInputMessageContentImage(BaseModel):
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


@json_schema_type
class OpenAIResponseInputMessage(BaseModel):
    content: str | list[OpenAIResponseInputMessageContent]
    role: Literal["system"] | Literal["developer"] | Literal["user"] | Literal["assistant"]
    type: Literal["message"] | None = "message"


@json_schema_type
class OpenAIResponseInputToolWebSearch(BaseModel):
    type: Literal["web_search"] | Literal["web_search_preview_2025_03_11"] = "web_search"
    # TODO: actually use search_context_size somewhere...
    search_context_size: str | None = Field(default="medium", pattern="^low|medium|high$")
    # TODO: add user_location


OpenAIResponseInputTool = Annotated[
    OpenAIResponseInputToolWebSearch,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputTool, name="OpenAIResponseInputTool")
