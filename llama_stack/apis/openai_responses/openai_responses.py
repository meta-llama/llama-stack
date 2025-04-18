# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncIterator, List, Literal, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


@json_schema_type
class OpenAIResponseError(BaseModel):
    code: str
    message: str


@json_schema_type
class OpenAIResponseOutputMessageContentOutputText(BaseModel):
    text: str
    type: Literal["output_text"] = "output_text"


OpenAIResponseOutputMessageContent = Annotated[
    Union[OpenAIResponseOutputMessageContentOutputText,],
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutputMessageContent, name="OpenAIResponseOutputMessageContent")


@json_schema_type
class OpenAIResponseOutputMessage(BaseModel):
    id: str
    content: List[OpenAIResponseOutputMessageContent]
    role: Literal["assistant"] = "assistant"
    status: str
    type: Literal["message"] = "message"


@json_schema_type
class OpenAIResponseOutputMessageWebSearchToolCall(BaseModel):
    id: str
    status: str
    type: Literal["web_search_call"] = "web_search_call"


OpenAIResponseOutput = Annotated[
    Union[
        OpenAIResponseOutputMessage,
        OpenAIResponseOutputMessageWebSearchToolCall,
    ],
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutput, name="OpenAIResponseOutput")


@json_schema_type
class OpenAIResponseObject(BaseModel):
    created_at: int
    error: Optional[OpenAIResponseError] = None
    id: str
    model: str
    object: Literal["response"] = "response"
    output: List[OpenAIResponseOutput]
    parallel_tool_calls: bool = False
    previous_response_id: Optional[str] = None
    status: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    truncation: Optional[str] = None
    user: Optional[str] = None


@json_schema_type
class OpenAIResponseObjectStream(BaseModel):
    response: OpenAIResponseObject
    type: Literal["response.created"] = "response.created"


@json_schema_type
class OpenAIResponseInputMessageContentText(BaseModel):
    text: str
    type: Literal["input_text"] = "input_text"


@json_schema_type
class OpenAIResponseInputMessageContentImage(BaseModel):
    detail: Literal["low", "high", "auto"] = "auto"
    type: Literal["input_image"] = "input_image"
    # TODO: handle file_id
    image_url: Optional[str] = None


# TODO: handle file content types
OpenAIResponseInputMessageContent = Annotated[
    Union[OpenAIResponseInputMessageContentText, OpenAIResponseInputMessageContentImage],
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputMessageContent, name="OpenAIResponseInputMessageContent")


@json_schema_type
class OpenAIResponseInputMessage(BaseModel):
    content: Union[str, List[OpenAIResponseInputMessageContent]]
    role: Literal["system", "developer", "user", "assistant"]
    type: Optional[Literal["message"]] = "message"


@json_schema_type
class OpenAIResponseInputToolWebSearch(BaseModel):
    type: Literal["web_search", "web_search_preview_2025_03_11"] = "web_search"
    search_context_size: Optional[str] = Field(default="medium", pattern="^low|medium|high$")
    # TODO: add user_location


OpenAIResponseInputTool = Annotated[
    Union[OpenAIResponseInputToolWebSearch,],
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputTool, name="OpenAIResponseInputTool")


@runtime_checkable
class OpenAIResponses(Protocol):
    """
    OpenAI Responses API implementation.
    """

    @webmethod(route="/openai/v1/responses/{id}", method="GET")
    async def get_openai_response(
        self,
        id: str,
    ) -> OpenAIResponseObject: ...

    @webmethod(route="/openai/v1/responses", method="POST")
    async def create_openai_response(
        self,
        input: Union[str, List[OpenAIResponseInputMessage]],
        model: str,
        previous_response_id: Optional[str] = None,
        store: Optional[bool] = True,
        stream: Optional[bool] = False,
        tools: Optional[List[OpenAIResponseInputTool]] = None,
    ) -> Union[OpenAIResponseObject, AsyncIterator[OpenAIResponseObjectStream]]: ...
