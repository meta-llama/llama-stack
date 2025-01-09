# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from pydantic import TypeAdapter

from llama_stack.apis.tools import (
    MCPConfig,
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from .config import ModelContextProtocolConfig


class ModelContextProtocolToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime):
    def __init__(self, config: ModelContextProtocolConfig):
        self.config = config

    async def initialize(self):
        pass

    async def list_runtime_tools(
        self,
        tool_group_id: Optional[str] = None,
        mcp_config: Optional[MCPConfig] = None,
    ) -> List[ToolDef]:
        if mcp_config is None:
            raise ValueError("mcp_config is required")

        tools = []
        async with stdio_client(
            StdioServerParameters(
                command=mcp_config.command,
                args=mcp_config.args,
                env=mcp_config.env,
            )
        ) as streams:
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
                        ToolDef(
                            name=tool.name,
                            description=tool.description,
                            parameters=parameters,
                            metadata={
                                "mcp_config": mcp_config.model_dump_json(),
                            },
                        )
                    )
        return tools

    async def invoke_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("mcp_config") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        mcp_config_dict = json.loads(tool.metadata.get("mcp_config"))
        mcp_config = TypeAdapter(MCPConfig).validate_python(mcp_config_dict)
        async with stdio_client(
            StdioServerParameters(
                command=mcp_config.command,
                args=mcp_config.args,
                env=mcp_config.env,
            )
        ) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool.identifier, arguments=args)

        return ToolInvocationResult(
            content="\n".join([result.model_dump_json() for result in result.content]),
            error_code=1 if result.isError else 0,
        )
