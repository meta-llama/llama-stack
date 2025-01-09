# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from llama_models.llama3.api.datatypes import ToolPromptFormat
from llama_models.schema_utils import json_schema_type, webmethod
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
    required: bool = Field(default=True)
    default: Optional[Any] = None


@json_schema_type
class ToolHost(Enum):
    distribution = "distribution"
    client = "client"
    model_context_protocol = "model_context_protocol"


@json_schema_type
class Tool(Resource):
    type: Literal[ResourceType.tool.value] = ResourceType.tool.value
    toolgroup_id: str
    tool_host: ToolHost
    description: str
    parameters: List[ToolParameter]
    metadata: Optional[Dict[str, Any]] = None
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )


@json_schema_type
class ToolDef(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[List[ToolParameter]] = None
    metadata: Optional[Dict[str, Any]] = None
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )


@json_schema_type
class ToolGroupInput(BaseModel):
    toolgroup_id: str
    provider_id: str
    args: Optional[Dict[str, Any]] = None
    mcp_endpoint: Optional[URL] = None


@json_schema_type
class ToolGroup(Resource):
    type: Literal[ResourceType.tool_group.value] = ResourceType.tool_group.value
    mcp_endpoint: Optional[URL] = None
    args: Optional[Dict[str, Any]] = None


@json_schema_type
class ToolInvocationResult(BaseModel):
    content: InterleavedContent
    error_message: Optional[str] = None
    error_code: Optional[int] = None


class ToolStore(Protocol):
    def get_tool(self, tool_name: str) -> Tool: ...
    def get_tool_group(self, tool_group_id: str) -> ToolGroup: ...


@runtime_checkable
@trace_protocol
class ToolGroups(Protocol):
    @webmethod(route="/toolgroups/register", method="POST")
    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: Optional[URL] = None,
        args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a tool group"""
        ...

    @webmethod(route="/toolgroups/get", method="GET")
    async def get_tool_group(
        self,
        toolgroup_id: str,
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

    # TODO: This needs to be renamed once OPEN API generator name conflict issue is fixed.
    @webmethod(route="/tool-runtime/list-tools", method="GET")
    async def list_runtime_tools(
        self, tool_group_id: Optional[str] = None, mcp_endpoint: Optional[URL] = None
    ) -> List[ToolDef]: ...

    @webmethod(route="/tool-runtime/invoke", method="POST")
    async def invoke_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolInvocationResult:
        """Run a tool with the given arguments"""
        ...
