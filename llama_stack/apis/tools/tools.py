# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field
from typing_extensions import runtime_checkable

from llama_stack.apis.common.content_types import URL, InterleavedContent
from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol
from llama_stack.schema_utils import json_schema_type, webmethod

from .rag_tool import RAGToolRuntime


@json_schema_type
class ToolParameter(BaseModel):
    name: str
    parameter_type: str
    description: str
    required: bool = Field(default=True)
    default: Any | None = None


@json_schema_type
class Tool(Resource):
    type: Literal[ResourceType.tool] = ResourceType.tool
    toolgroup_id: str
    description: str
    parameters: list[ToolParameter]
    metadata: dict[str, Any] | None = None


@json_schema_type
class ToolDef(BaseModel):
    name: str
    description: str | None = None
    parameters: list[ToolParameter] | None = None
    metadata: dict[str, Any] | None = None


@json_schema_type
class ToolGroupInput(BaseModel):
    toolgroup_id: str
    provider_id: str
    args: dict[str, Any] | None = None
    mcp_endpoint: URL | None = None


@json_schema_type
class ToolGroup(Resource):
    type: Literal[ResourceType.tool_group] = ResourceType.tool_group
    mcp_endpoint: URL | None = None
    args: dict[str, Any] | None = None


@json_schema_type
class ToolInvocationResult(BaseModel):
    content: InterleavedContent | None = None
    error_message: str | None = None
    error_code: int | None = None
    metadata: dict[str, Any] | None = None


class ToolStore(Protocol):
    async def get_tool(self, tool_name: str) -> Tool: ...
    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup: ...


class ListToolGroupsResponse(BaseModel):
    data: list[ToolGroup]


class ListToolsResponse(BaseModel):
    data: list[Tool]


class ListToolDefsResponse(BaseModel):
    data: list[ToolDef]


@runtime_checkable
@trace_protocol
class ToolGroups(Protocol):
    @webmethod(route="/toolgroups", method="POST")
    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: URL | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool group.

        :param toolgroup_id: The ID of the tool group to register.
        :param provider_id: The ID of the provider to use for the tool group.
        :param mcp_endpoint: The MCP endpoint to use for the tool group.
        :param args: A dictionary of arguments to pass to the tool group.
        """
        ...

    @webmethod(route="/toolgroups/{toolgroup_id:path}", method="GET")
    async def get_tool_group(
        self,
        toolgroup_id: str,
    ) -> ToolGroup:
        """Get a tool group by its ID.

        :param toolgroup_id: The ID of the tool group to get.
        :returns: A ToolGroup.
        """
        ...

    @webmethod(route="/toolgroups", method="GET")
    async def list_tool_groups(self) -> ListToolGroupsResponse:
        """List tool groups with optional provider.

        :returns: A ListToolGroupsResponse.
        """
        ...

    @webmethod(route="/tools", method="GET")
    async def list_tools(self, toolgroup_id: str | None = None) -> ListToolsResponse:
        """List tools with optional tool group.

        :param toolgroup_id: The ID of the tool group to list tools for.
        :returns: A ListToolsResponse.
        """
        ...

    @webmethod(route="/tools/{tool_name:path}", method="GET")
    async def get_tool(
        self,
        tool_name: str,
    ) -> Tool:
        """Get a tool by its name.

        :param tool_name: The name of the tool to get.
        :returns: A Tool.
        """
        ...

    @webmethod(route="/toolgroups/{toolgroup_id:path}", method="DELETE")
    async def unregister_toolgroup(
        self,
        toolgroup_id: str,
    ) -> None:
        """Unregister a tool group.

        :param toolgroup_id: The ID of the tool group to unregister.
        """
        ...


class SpecialToolGroup(Enum):
    rag_tool = "rag_tool"


@runtime_checkable
@trace_protocol
class ToolRuntime(Protocol):
    tool_store: ToolStore | None = None

    rag_tool: RAGToolRuntime | None = None

    # TODO: This needs to be renamed once OPEN API generator name conflict issue is fixed.
    @webmethod(route="/tool-runtime/list-tools", method="GET")
    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        """List all tools in the runtime.

        :param tool_group_id: The ID of the tool group to list tools for.
        :param mcp_endpoint: The MCP endpoint to use for the tool group.
        :returns: A ListToolDefsResponse.
        """
        ...

    @webmethod(route="/tool-runtime/invoke", method="POST")
    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        """Run a tool with the given arguments.

        :param tool_name: The name of the tool to invoke.
        :param kwargs: A dictionary of arguments to pass to the tool.
        :returns: A ToolInvocationResult.
        """
        ...
