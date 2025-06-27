# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from llama_stack.apis.vector_io import SearchRankingOptions as FileSearchRankingOptions
from llama_stack.schema_utils import json_schema_type, register_schema

# NOTE(ashwin): this file is literally a copy of the OpenAI responses API schema. We should probably
# take their YAML and generate this file automatically. Their YAML is available.


@json_schema_type
class OpenAIResponseError(BaseModel):
    code: str
    message: str


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
class OpenAIResponseAnnotationFileCitation(BaseModel):
    type: Literal["file_citation"] = "file_citation"
    file_id: str
    filename: str
    index: int


@json_schema_type
class OpenAIResponseAnnotationCitation(BaseModel):
    type: Literal["url_citation"] = "url_citation"
    end_index: int
    start_index: int
    title: str
    url: str


@json_schema_type
class OpenAIResponseAnnotationContainerFileCitation(BaseModel):
    type: Literal["container_file_citation"] = "container_file_citation"
    container_id: str
    end_index: int
    file_id: str
    filename: str
    start_index: int


@json_schema_type
class OpenAIResponseAnnotationFilePath(BaseModel):
    type: Literal["file_path"] = "file_path"
    file_id: str
    index: int


OpenAIResponseAnnotations = Annotated[
    OpenAIResponseAnnotationFileCitation
    | OpenAIResponseAnnotationCitation
    | OpenAIResponseAnnotationContainerFileCitation
    | OpenAIResponseAnnotationFilePath,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseAnnotations, name="OpenAIResponseAnnotations")


@json_schema_type
class OpenAIResponseOutputMessageContentOutputText(BaseModel):
    text: str
    type: Literal["output_text"] = "output_text"
    annotations: list[OpenAIResponseAnnotations] = Field(default_factory=list)


OpenAIResponseOutputMessageContent = Annotated[
    OpenAIResponseOutputMessageContentOutputText,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutputMessageContent, name="OpenAIResponseOutputMessageContent")


@json_schema_type
class OpenAIResponseMessage(BaseModel):
    """
    Corresponds to the various Message types in the Responses API.
    They are all under one type because the Responses API gives them all
    the same "type" value, and there is no way to tell them apart in certain
    scenarios.
    """

    content: str | list[OpenAIResponseInputMessageContent] | list[OpenAIResponseOutputMessageContent]
    role: Literal["system"] | Literal["developer"] | Literal["user"] | Literal["assistant"]
    type: Literal["message"] = "message"

    # The fields below are not used in all scenarios, but are required in others.
    id: str | None = None
    status: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageWebSearchToolCall(BaseModel):
    id: str
    status: str
    type: Literal["web_search_call"] = "web_search_call"


@json_schema_type
class OpenAIResponseOutputMessageFileSearchToolCall(BaseModel):
    id: str
    queries: list[str]
    status: str
    type: Literal["file_search_call"] = "file_search_call"
    results: list[dict[str, Any]] | None = None


@json_schema_type
class OpenAIResponseOutputMessageFunctionToolCall(BaseModel):
    call_id: str
    name: str
    arguments: str
    type: Literal["function_call"] = "function_call"
    id: str | None = None
    status: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageMCPCall(BaseModel):
    id: str
    type: Literal["mcp_call"] = "mcp_call"
    arguments: str
    name: str
    server_label: str
    error: str | None = None
    output: str | None = None


class MCPListToolsTool(BaseModel):
    input_schema: dict[str, Any]
    name: str
    description: str | None = None


@json_schema_type
class OpenAIResponseOutputMessageMCPListTools(BaseModel):
    id: str
    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    server_label: str
    tools: list[MCPListToolsTool]


OpenAIResponseOutput = Annotated[
    OpenAIResponseMessage
    | OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseOutputMessageMCPCall
    | OpenAIResponseOutputMessageMCPListTools,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseOutput, name="OpenAIResponseOutput")


# This has to be a TypedDict because we need a "schema" field and our strong
# typing code in the schema generator doesn't support Pydantic aliases. That also
# means we can't use a discriminator field here, because TypedDicts don't support
# default values which the strong typing code requires for discriminators.
class OpenAIResponseTextFormat(TypedDict, total=False):
    """Configuration for Responses API text format.

    :param type: Must be "text", "json_schema", or "json_object" to identify the format type
    :param name: The name of the response format. Only used for json_schema.
    :param schema: The JSON schema the response should conform to. In a Python SDK, this is often a `pydantic` model. Only used for json_schema.
    :param description: (Optional) A description of the response format. Only used for json_schema.
    :param strict: (Optional) Whether to strictly enforce the JSON schema. If true, the response must match the schema exactly. Only used for json_schema.
    """

    type: Literal["text"] | Literal["json_schema"] | Literal["json_object"]
    name: str | None
    schema: dict[str, Any] | None
    description: str | None
    strict: bool | None


@json_schema_type
class OpenAIResponseText(BaseModel):
    format: OpenAIResponseTextFormat | None = None


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
    # Default to text format to avoid breaking the loading of old responses
    # before the field was added. New responses will have this set always.
    text: OpenAIResponseText = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text"))
    top_p: float | None = None
    truncation: str | None = None
    user: str | None = None


@json_schema_type
class OpenAIDeleteResponseObject(BaseModel):
    id: str
    object: Literal["response"] = "response"
    deleted: bool = True


@json_schema_type
class OpenAIResponseObjectStreamResponseCreated(BaseModel):
    response: OpenAIResponseObject
    type: Literal["response.created"] = "response.created"


@json_schema_type
class OpenAIResponseObjectStreamResponseCompleted(BaseModel):
    response: OpenAIResponseObject
    type: Literal["response.completed"] = "response.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputItemAdded(BaseModel):
    response_id: str
    item: OpenAIResponseOutput
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.added"] = "response.output_item.added"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputItemDone(BaseModel):
    response_id: str
    item: OpenAIResponseOutput
    output_index: int
    sequence_number: int
    type: Literal["response.output_item.done"] = "response.output_item.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextDelta(BaseModel):
    content_index: int
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.delta"] = "response.output_text.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseOutputTextDone(BaseModel):
    content_index: int
    text: str  # final text of the output item
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.output_text.done"] = "response.output_text.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta(BaseModel):
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone(BaseModel):
    arguments: str  # final arguments of the function call
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallInProgress(BaseModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.in_progress"] = "response.web_search_call.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallSearching(BaseModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.searching"] = "response.web_search_call.searching"


@json_schema_type
class OpenAIResponseObjectStreamResponseWebSearchCallCompleted(BaseModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.web_search_call.completed"] = "response.web_search_call.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsInProgress(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.in_progress"] = "response.mcp_list_tools.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsFailed(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.failed"] = "response.mcp_list_tools.failed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpListToolsCompleted(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_list_tools.completed"] = "response.mcp_list_tools.completed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta(BaseModel):
    delta: str
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.arguments.delta"] = "response.mcp_call.arguments.delta"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallArgumentsDone(BaseModel):
    arguments: str  # final arguments of the MCP call
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.arguments.done"] = "response.mcp_call.arguments.done"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallInProgress(BaseModel):
    item_id: str
    output_index: int
    sequence_number: int
    type: Literal["response.mcp_call.in_progress"] = "response.mcp_call.in_progress"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallFailed(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_call.failed"] = "response.mcp_call.failed"


@json_schema_type
class OpenAIResponseObjectStreamResponseMcpCallCompleted(BaseModel):
    sequence_number: int
    type: Literal["response.mcp_call.completed"] = "response.mcp_call.completed"


OpenAIResponseObjectStream = Annotated[
    OpenAIResponseObjectStreamResponseCreated
    | OpenAIResponseObjectStreamResponseOutputItemAdded
    | OpenAIResponseObjectStreamResponseOutputItemDone
    | OpenAIResponseObjectStreamResponseOutputTextDelta
    | OpenAIResponseObjectStreamResponseOutputTextDone
    | OpenAIResponseObjectStreamResponseFunctionCallArgumentsDelta
    | OpenAIResponseObjectStreamResponseFunctionCallArgumentsDone
    | OpenAIResponseObjectStreamResponseWebSearchCallInProgress
    | OpenAIResponseObjectStreamResponseWebSearchCallSearching
    | OpenAIResponseObjectStreamResponseWebSearchCallCompleted
    | OpenAIResponseObjectStreamResponseMcpListToolsInProgress
    | OpenAIResponseObjectStreamResponseMcpListToolsFailed
    | OpenAIResponseObjectStreamResponseMcpListToolsCompleted
    | OpenAIResponseObjectStreamResponseMcpCallArgumentsDelta
    | OpenAIResponseObjectStreamResponseMcpCallArgumentsDone
    | OpenAIResponseObjectStreamResponseMcpCallInProgress
    | OpenAIResponseObjectStreamResponseMcpCallFailed
    | OpenAIResponseObjectStreamResponseMcpCallCompleted
    | OpenAIResponseObjectStreamResponseCompleted,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseObjectStream, name="OpenAIResponseObjectStream")


@json_schema_type
class OpenAIResponseInputFunctionToolCallOutput(BaseModel):
    """
    This represents the output of a function call that gets passed back to the model.
    """

    call_id: str
    output: str
    type: Literal["function_call_output"] = "function_call_output"
    id: str | None = None
    status: str | None = None


OpenAIResponseInput = Annotated[
    # Responses API allows output messages to be passed in as input
    OpenAIResponseOutputMessageWebSearchToolCall
    | OpenAIResponseOutputMessageFileSearchToolCall
    | OpenAIResponseOutputMessageFunctionToolCall
    | OpenAIResponseInputFunctionToolCallOutput
    |
    # Fallback to the generic message type as a last resort
    OpenAIResponseMessage,
    Field(union_mode="left_to_right"),
]
register_schema(OpenAIResponseInput, name="OpenAIResponseInput")


# Must match type Literals of OpenAIResponseInputToolWebSearch below
WebSearchToolTypes = ["web_search", "web_search_preview", "web_search_preview_2025_03_11"]


@json_schema_type
class OpenAIResponseInputToolWebSearch(BaseModel):
    # Must match values of WebSearchToolTypes above
    type: Literal["web_search"] | Literal["web_search_preview"] | Literal["web_search_preview_2025_03_11"] = (
        "web_search"
    )
    # TODO: actually use search_context_size somewhere...
    search_context_size: str | None = Field(default="medium", pattern="^low|medium|high$")
    # TODO: add user_location


@json_schema_type
class OpenAIResponseInputToolFunction(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None
    strict: bool | None = None


@json_schema_type
class OpenAIResponseInputToolFileSearch(BaseModel):
    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str]
    filters: dict[str, Any] | None = None
    max_num_results: int | None = Field(default=10, ge=1, le=50)
    ranking_options: FileSearchRankingOptions | None = None


class ApprovalFilter(BaseModel):
    always: list[str] | None = None
    never: list[str] | None = None


class AllowedToolsFilter(BaseModel):
    tool_names: list[str] | None = None


@json_schema_type
class OpenAIResponseInputToolMCP(BaseModel):
    type: Literal["mcp"] = "mcp"
    server_label: str
    server_url: str
    headers: dict[str, Any] | None = None

    require_approval: Literal["always"] | Literal["never"] | ApprovalFilter = "never"
    allowed_tools: list[str] | AllowedToolsFilter | None = None


OpenAIResponseInputTool = Annotated[
    OpenAIResponseInputToolWebSearch
    | OpenAIResponseInputToolFileSearch
    | OpenAIResponseInputToolFunction
    | OpenAIResponseInputToolMCP,
    Field(discriminator="type"),
]
register_schema(OpenAIResponseInputTool, name="OpenAIResponseInputTool")


class ListOpenAIResponseInputItem(BaseModel):
    data: list[OpenAIResponseInput]
    object: Literal["list"] = "list"


@json_schema_type
class OpenAIResponseObjectWithInput(OpenAIResponseObject):
    input: list[OpenAIResponseInput]


@json_schema_type
class ListOpenAIResponseObject(BaseModel):
    data: list[OpenAIResponseObjectWithInput]
    has_more: bool
    first_id: str
    last_id: str
    object: Literal["list"] = "list"
