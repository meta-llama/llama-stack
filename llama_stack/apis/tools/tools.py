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

from llama_stack.apis.common.content_types import InterleavedContent, URL
from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol


@json_schema_type
class ToolParameter(BaseModel):
    name: str
    parameter_type: str
    description: str
    required: bool
    default: Optional[Any] = None


@json_schema_type
class Tool(Resource):
    type: Literal[ResourceType.tool.value] = ResourceType.tool.value
    tool_group: str
    description: str
    parameters: List[ToolParameter]
    provider_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
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
class MCPToolGroupDef(BaseModel):
    """
    A tool group that is defined by in a model context protocol server.
    Refer to https://modelcontextprotocol.io/docs/concepts/tools for more information.
    """

    type: Literal["model_context_protocol"] = "model_context_protocol"
    endpoint: URL


@json_schema_type
class UserDefinedToolGroupDef(BaseModel):
    type: Literal["user_defined"] = "user_defined"
    tools: List[ToolDef]


ToolGroupDef = register_schema(
    Annotated[
        Union[MCPToolGroupDef, UserDefinedToolGroupDef], Field(discriminator="type")
    ],
    name="ToolGroupDef",
)


class ToolGroupInput(BaseModel):
    tool_group_id: str
    tool_group: ToolGroupDef
    provider_id: Optional[str] = None


class ToolGroup(Resource):
    type: Literal[ResourceType.tool_group.value] = ResourceType.tool_group.value


@json_schema_type
class ToolInvocationResult(BaseModel):
    content: InterleavedContent
    error_message: Optional[str] = None
    error_code: Optional[int] = None


class ToolStore(Protocol):
    def get_tool(self, tool_name: str) -> Tool: ...


@runtime_checkable
@trace_protocol
class ToolGroups(Protocol):
    @webmethod(route="/toolgroups/register", method="POST")
    async def register_tool_group(
        self,
        tool_group_id: str,
        tool_group: ToolGroupDef,
        provider_id: Optional[str] = None,
    ) -> None:
        """Register a tool group"""
        ...

    @webmethod(route="/toolgroups/get", method="GET")
    async def get_tool_group(
        self,
        tool_group_id: str,
    ) -> ToolGroup: ...

    @webmethod(route="/toolgroups/list", method="GET")
    async def list_tool_groups(self) -> List[ToolGroup]:
        """List tool groups with optional provider"""
        ...

    @webmethod(route="/tools/list", method="GET")
    async def list_tools(self, tool_group_id: Optional[str] = None) -> List[Tool]:
        """List tools with optional tool group"""
        ...

    @webmethod(route="/tools/get", method="GET")
    async def get_tool(self, tool_name: str) -> Tool: ...

    @webmethod(route="/toolgroups/unregister", method="POST")
    async def unregister_tool_group(self, tool_group_id: str) -> None:
        """Unregister a tool group"""
        ...


@runtime_checkable
@trace_protocol
class ToolRuntime(Protocol):
    tool_store: ToolStore

    @webmethod(route="/tool-runtime/discover", method="POST")
    async def discover_tools(self, tool_group: ToolGroupDef) -> List[ToolDef]: ...

    @webmethod(route="/tool-runtime/invoke", method="POST")
    async def invoke_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolInvocationResult:
        """Run a tool with the given arguments"""
        ...
