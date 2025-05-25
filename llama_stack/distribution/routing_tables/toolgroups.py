# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.tools import ListToolGroupsResponse, ListToolsResponse, Tool, ToolGroup, ToolGroups
from llama_stack.distribution.datatypes import ToolGroupWithACL
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core")


class ToolGroupsRoutingTable(CommonRoutingTableImpl, ToolGroups):
    toolgroups_to_tools: dict[str, list[Tool]] = {}
    tool_to_toolgroup: dict[str, str] = {}

    # overridden
    def get_provider_impl(self, routing_key: str, provider_id: str | None = None) -> Any:
        # we don't index tools in the registry anymore, but only keep a cache of them by toolgroup_id
        # TODO: we may want to invalidate the cache (for a given toolgroup_id) every once in a while?
        tool_name = routing_key
        if tool_name in self.tool_to_toolgroup:
            routing_key = self.tool_to_toolgroup[tool_name]
        return super().get_provider_impl(routing_key, provider_id)

    async def list_tools(self, toolgroup_id: str | None = None) -> ListToolsResponse:
        if toolgroup_id:
            toolgroups = [await self.get_tool_group(toolgroup_id)]
        else:
            toolgroups = await self.get_all_with_type("tool_group")

        all_tools = []
        for toolgroup in toolgroups:
            group_id = toolgroup.identifier
            if group_id not in self.toolgroups_to_tools:
                provider_impl = self.get_provider_impl(toolgroup.provider_id)
                tools = await provider_impl.list_runtime_tools(group_id, toolgroup.mcp_endpoint)

                self.toolgroups_to_tools[group_id] = tools.data
                for tool in tools.data:
                    self.tool_to_toolgroup[tool.identifier] = group_id
            all_tools.extend(self.toolgroups_to_tools[group_id])

        return ListToolsResponse(data=all_tools)

    async def list_tool_groups(self) -> ListToolGroupsResponse:
        return ListToolGroupsResponse(data=await self.get_all_with_type("tool_group"))

    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup:
        tool_group = await self.get_object_by_identifier("tool_group", toolgroup_id)
        if tool_group is None:
            raise ValueError(f"Tool group '{toolgroup_id}' not found")
        return tool_group

    async def get_tool(self, tool_name: str) -> Tool:
        if tool_name in self.tool_to_toolgroup:
            toolgroup_id = self.tool_to_toolgroup[tool_name]
            tools = self.toolgroups_to_tools[toolgroup_id]
            for tool in tools:
                if tool.identifier == tool_name:
                    return tool
        raise ValueError(f"Tool '{tool_name}' not found")

    async def register_tool_group(
        self,
        toolgroup_id: str,
        provider_id: str,
        mcp_endpoint: URL | None = None,
        args: dict[str, Any] | None = None,
    ) -> None:
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
        await self.unregister_object(tool_group)

    async def shutdown(self) -> None:
        pass
