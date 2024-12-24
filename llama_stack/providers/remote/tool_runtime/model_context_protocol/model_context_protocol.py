# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List
from urllib.parse import urlparse

from llama_stack.apis.tools import (
    CustomToolDef,
    MCPToolGroupDef,
    ToolDef,
    ToolGroupDef,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from mcp import ClientSession
from mcp.client.sse import sse_client

from .config import ModelContextProtocolConfig


class ModelContextProtocolToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime):
    def __init__(self, config: ModelContextProtocolConfig):
        self.config = config

    async def initialize(self):
        pass

    async def discover_tools(self, tool_group: ToolGroupDef) -> List[ToolDef]:
        if not isinstance(tool_group, MCPToolGroupDef):
            raise ValueError(f"Unsupported tool group type: {type(tool_group)}")

        tools = []
        async with sse_client(tool_group.endpoint.uri) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    parameters = []
                    for param_name, param_schema in tool.inputSchema.get(
                        "properties", {}
                    ).items():
                        parameters.append(
                            ToolParameter(
                                name=param_name,
                                parameter_type=param_schema.get("type", "string"),
                                description=param_schema.get("description", ""),
                            )
                        )
                    tools.append(
                        CustomToolDef(
                            name=tool.name,
                            description=tool.description,
                            parameters=parameters,
                            metadata={
                                "endpoint": tool_group.endpoint.uri,
                            },
                        )
                    )
        return tools

    async def invoke_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint = tool.metadata.get("endpoint")
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        async with sse_client(endpoint) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool.identifier, args)

        return ToolInvocationResult(
            content="\n".join([result.model_dump_json() for result in result.content]),
            error_code=1 if result.isError else 0,
        )
