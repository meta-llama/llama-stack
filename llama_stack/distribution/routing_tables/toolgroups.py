# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.tools import ListToolGroupsResponse, ListToolsResponse, Tool, ToolGroup, ToolGroups
from llama_stack.distribution.datatypes import ToolGroupWithOwner
from llama_stack.log import get_logger

from .common import CommonRoutingTableImpl

logger = get_logger(name=__name__, category="core")


def parse_toolgroup_from_toolgroup_name_pair(toolgroup_name_with_maybe_tool_name: str) -> str | None:
    # handle the funny case like "builtin::rag/knowledge_search"
    parts = toolgroup_name_with_maybe_tool_name.split("/")
    if len(parts) == 2:
        return parts[0]
    else:
        return None


class ToolGroupsRoutingTable(CommonRoutingTableImpl, ToolGroups):
    toolgroups_to_tools: dict[str, list[Tool]] = {}
    tool_to_toolgroup: dict[str, str] = {}

    # overridden
    async def get_provider_impl(self, routing_key: str, provider_id: str | None = None) -> Any:
        # we don't index tools in the registry anymore, but only keep a cache of them by toolgroup_id
        # TODO: we may want to invalidate the cache (for a given toolgroup_id) every once in a while?

        toolgroup_id = parse_toolgroup_from_toolgroup_name_pair(routing_key)
        if toolgroup_id:
            routing_key = toolgroup_id

        if routing_key in self.tool_to_toolgroup:
            routing_key = self.tool_to_toolgroup[routing_key]
        return await super().get_provider_impl(routing_key, provider_id)

    async def list_tools(self, toolgroup_id: str | None = None) -> ListToolsResponse:
        if toolgroup_id:
            if group_id := parse_toolgroup_from_toolgroup_name_pair(toolgroup_id):
                toolgroup_id = group_id
            toolgroups = [await self.get_tool_group(toolgroup_id)]
        else:
            toolgroups = await self.get_all_with_type("tool_group")

        all_tools = []
        for toolgroup in toolgroups:
            if toolgroup.identifier not in self.toolgroups_to_tools:
                await self._index_tools(toolgroup)
            all_tools.extend(self.toolgroups_to_tools[toolgroup.identifier])

        return ListToolsResponse(data=all_tools)

    async def _index_tools(self, toolgroup: ToolGroup):
        provider_impl = await super().get_provider_impl(toolgroup.identifier, toolgroup.provider_id)
        tooldefs_response = await provider_impl.list_runtime_tools(toolgroup.identifier, toolgroup.mcp_endpoint)

        # TODO: kill this Tool vs ToolDef distinction
        tooldefs = tooldefs_response.data
        tools = []
        for t in tooldefs:
            tools.append(
                Tool(
                    identifier=t.name,
                    toolgroup_id=toolgroup.identifier,
                    description=t.description or "",
                    parameters=t.parameters or [],
                    metadata=t.metadata,
                    provider_id=toolgroup.provider_id,
                    annotations=t.annotations,
                )
            )

        self.toolgroups_to_tools[toolgroup.identifier] = tools
        for tool in tools:
            self.tool_to_toolgroup[tool.identifier] = toolgroup.identifier

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
        toolgroup = ToolGroupWithOwner(
            identifier=toolgroup_id,
            provider_id=provider_id,
            provider_resource_id=toolgroup_id,
            mcp_endpoint=mcp_endpoint,
            args=args,
        )
        await self.register_object(toolgroup)

        # ideally, indexing of the tools should not be necessary because anyone using
        # the tools should first list the tools and then use them. but there are assumptions
        # baked in some of the code and tests right now.
        if not toolgroup.mcp_endpoint:
            await self._index_tools(toolgroup)
        return toolgroup

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        tool_group = await self.get_tool_group(toolgroup_id)
        if tool_group is None:
            raise ValueError(f"Tool group {toolgroup_id} not found")
        await self.unregister_object(tool_group)

    async def shutdown(self) -> None:
        pass
