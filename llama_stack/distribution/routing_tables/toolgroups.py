# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.tools import ListToolGroupsResponse, ListToolsResponse, Tool, ToolGroup, ToolGroups, ToolHost
from llama_stack.distribution.datatypes import (
    ToolGroupWithACL,
    ToolWithACL,
)
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core")


class ToolGroupsRoutingTable(CommonRoutingTableImpl, ToolGroups):
    async def list_tools(self, toolgroup_id: str | None = None) -> ListToolsResponse:
        tools = await self.get_all_with_type("tool")
        if toolgroup_id:
            tools = [tool for tool in tools if tool.toolgroup_id == toolgroup_id]
        return ListToolsResponse(data=tools)

    async def list_tool_groups(self) -> ListToolGroupsResponse:
        return ListToolGroupsResponse(data=await self.get_all_with_type("tool_group"))

    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup:
        tool_group = await self.get_object_by_identifier("tool_group", toolgroup_id)
        if tool_group is None:
            raise ValueError(f"Tool group '{toolgroup_id}' not found")
        return tool_group

    async def get_tool(self, tool_name: str) -> Tool:
        return await self.get_object_by_identifier("tool", tool_name)

    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: URL | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
        tools = []
        tool_defs = await self.impls_by_provider_id[provider_id].list_runtime_tools(toolgroup_id, mcp_endpoint)
        tool_host = ToolHost.model_context_protocol if mcp_endpoint else ToolHost.distribution

        for tool_def in tool_defs.data:
            tools.append(
                ToolWithACL(
                    identifier=tool_def.name,
                    toolgroup_id=toolgroup_id,
                    description=tool_def.description or "",
                    parameters=tool_def.parameters or [],
                    provider_id=provider_id,
                    provider_resource_id=tool_def.name,
                    metadata=tool_def.metadata,
                    tool_host=tool_host,
                )
            )
        for tool in tools:
            existing_tool = await self.get_tool(tool.identifier)
            # Compare existing and new object if one exists
            if existing_tool:
                existing_dict = existing_tool.model_dump()
                new_dict = tool.model_dump()

                if existing_dict != new_dict:
                    raise ValueError(
                        f"Object {tool.identifier} already exists in registry. Please use a different identifier."
                    )
            await self.register_object(tool)

        await self.dist_registry.register(
            ToolGroupWithACL(
                identifier=toolgroup_id,
                provider_id=provider_id,
                provider_resource_id=toolgroup_id,
                mcp_endpoint=mcp_endpoint,
                args=args,
            )
        )

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        tool_group = await self.get_tool_group(toolgroup_id)
        if tool_group is None:
            raise ValueError(f"Tool group {toolgroup_id} not found")
        tools = await self.list_tools(toolgroup_id)
        for tool in getattr(tools, "data", []):
            await self.unregister_object(tool)
        await self.unregister_object(tool_group)

    async def shutdown(self) -> None:
        pass
