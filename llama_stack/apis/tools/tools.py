# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional

from llama_models.llama3.api.datatypes import ToolPromptFormat

from llama_models.schema_utils import json_schema_type
from pydantic import BaseModel, Field
from typing_extensions import Protocol, runtime_checkable

from llama_stack.apis.resource import Resource
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

    resource_type: Literal["tool"] = "tool"
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
    async def register_tool(
        self,
        tool_id: str,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        returns: ToolReturn,
        provider_metadata: Optional[Dict[str, Any]] = None,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
    ) -> Tool:
        """Register a tool with provider-specific metadata"""
        ...

    async def get_tool(
        self,
        identifier: str,
    ) -> Tool: ...

    async def list_tools(
        self,
        provider_id: Optional[str] = None,
    ) -> List[Tool]:
        """List tools with optional provider"""


@runtime_checkable
@trace_protocol
class ToolRuntime(Protocol):
    def invoke_tool(self, tool_id: str, args: Dict[str, Any]) -> Any:
        """Run a tool with the given arguments"""
        ...
