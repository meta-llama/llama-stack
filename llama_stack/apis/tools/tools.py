# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from llama_models.llama3.api.datatypes import ToolPromptFormat
from llama_models.schema_utils import json_schema_type, register_schema, webmethod
from pydantic import BaseModel, Field
from typing_extensions import Protocol, runtime_checkable

from llama_stack.apis.inference import InterleavedContent

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol


@json_schema_type
class ToolParameter(BaseModel):
    name: str
    parameter_type: str
    description: str


@json_schema_type
class Tool(Resource):
    type: Literal[ResourceType.tool.value] = ResourceType.tool.value
    name: str
    tool_group: str
    description: str
    parameters: List[ToolParameter]
    provider_id: Optional[str] = None
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )


@json_schema_type
class ToolDef(BaseModel):
    name: str
    description: str
    parameters: List[ToolParameter]
    metadata: Dict[str, Any]
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )


@json_schema_type
class MCPToolGroup(BaseModel):
    type: Literal["mcp"] = "mcp"
    endpoint: str


@json_schema_type
class UserDefinedToolGroup(BaseModel):
    type: Literal["user_defined"] = "user_defined"
    tools: List[ToolDef]


ToolGroup = register_schema(
    Annotated[Union[MCPToolGroup, UserDefinedToolGroup], Field(discriminator="type")],
    name="ToolGroup",
)


@json_schema_type
class InvokeToolResult(BaseModel):
    content: InterleavedContent
    error_message: Optional[str] = None
    error_code: Optional[int] = None


class ToolStore(Protocol):
    def get_tool(self, tool_id: str) -> Tool: ...


@runtime_checkable
@trace_protocol
class Tools(Protocol):
    @webmethod(route="/tool-groups/register", method="POST")
    async def register_tool_group(
        self,
        name: str,
        tool_group: ToolGroup,
        provider_id: Optional[str] = None,
    ) -> None:
        """Register a tool group"""
        ...

    @webmethod(route="/tools/get", method="GET")
    async def get_tool(
        self,
        tool_id: str,
    ) -> Tool: ...

    @webmethod(route="/tools/list", method="GET")
    async def list_tools(self) -> List[Tool]:
        """List tools with optional provider"""

    @webmethod(route="/tools/unregister", method="POST")
    async def unregister_tool(self, tool_id: str) -> None:
        """Unregister a tool"""
        ...


@runtime_checkable
@trace_protocol
class ToolRuntime(Protocol):
    tool_store: ToolStore

    @webmethod(route="/tool-runtime/invoke", method="POST")
    async def invoke_tool(self, tool_id: str, args: Dict[str, Any]) -> InvokeToolResult:
        """Run a tool with the given arguments"""
        ...
