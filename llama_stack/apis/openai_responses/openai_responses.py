# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncIterator, List, Literal, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_stack.schema_utils import json_schema_type, webmethod


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


@json_schema_type
class OpenAIResponseOutputMessage(BaseModel):
    id: str
    content: List[OpenAIResponseOutputMessageContent]
    role: Literal["assistant"] = "assistant"
    status: str
    type: Literal["message"] = "message"


OpenAIResponseOutput = Annotated[
    Union[OpenAIResponseOutputMessage,],
    Field(discriminator="type"),
]


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
        input: str,
        model: str,
        previous_response_id: Optional[str] = None,
        store: Optional[bool] = True,
        stream: Optional[bool] = False,
    ) -> Union[OpenAIResponseObject, AsyncIterator[OpenAIResponseObjectStream]]: ...
