# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional

from llama_models.llama3.api.datatypes import ToolPromptFormat
from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel, Field
from typing_extensions import Protocol, runtime_checkable

from llama_stack.apis.resource import Resource, ResourceType
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol


@json_schema_type
class ToolParameter(BaseModel):
    """Represents a parameter in a tool's function signature"""

    name: str
    type_hint: str
    description: str
    required: bool = True
    default: Optional[Any] = None


@json_schema_type
class ToolReturn(BaseModel):
    """Represents the return type and description of a tool"""

    type_hint: str
    description: str


@json_schema_type
class Tool(Resource):
    """Represents a tool that can be provided by different providers"""

    type: Literal[ResourceType.tool.value] = ResourceType.tool.value
    name: str
    description: str
    parameters: List[ToolParameter]
    returns: ToolReturn
    provider_metadata: Optional[Dict[str, Any]] = None
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )


@runtime_checkable
@trace_protocol
class Tools(Protocol):
    @webmethod(route="/tools/register", method="POST")
    async def register_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        returns: ToolReturn,
        provider_id: Optional[str] = None,
        provider_id: Optional[str] = None,
        provider_resource_id: Optional[str] = None,
        provider_metadata: Optional[Dict[str, Any]] = None,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
    ) -> Tool:
        """Register a tool with provider-specific metadata"""
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
    @webmethod(route="/tool-runtime/invoke", method="POST")
    async def invoke_tool(self, tool_id: str, args: Dict[str, Any]) -> Any:
        """Run a tool with the given arguments"""
        ...
