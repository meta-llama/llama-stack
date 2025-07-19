# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any
from urllib.parse import urlparse

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.datatypes import Api
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolGroup,
    ToolInvocationResult,
    ToolRuntime,
)
from llama_stack.distribution.datatypes import AuthenticationRequiredError
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ToolGroupsProtocolPrivate
from llama_stack.providers.utils.tools.mcp import invoke_mcp_tool, list_mcp_tools

from .config import MCPProviderConfig

logger = get_logger(__name__, category="tools")


class ModelContextProtocolToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    def __init__(self, config: MCPProviderConfig, _deps: dict[Api, Any]):
        self.config = config

    async def initialize(self):
        pass

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        # this endpoint should be retrieved by getting the tool group right?
        if mcp_endpoint is None:
            raise ValueError("mcp_endpoint is required")

        logger.debug(f"Listing runtime tools for toolgroup {tool_group_id} at endpoint {mcp_endpoint.uri}")
        headers = await self.get_headers_from_request(mcp_endpoint.uri)

        if headers:
            logger.debug(f"Found {len(headers)} headers for MCP endpoint {mcp_endpoint.uri}")
            # Log header keys but not values for security
            header_keys = list(headers.keys())
            logger.debug(f"Header keys: {header_keys}")
        else:
            logger.debug(f"No headers found for MCP endpoint {mcp_endpoint.uri}")

        try:
            result = await list_mcp_tools(mcp_endpoint.uri, headers)
            logger.info(f"Successfully listed {len(result.data)} tools for toolgroup {tool_group_id}")
            return result
        except AuthenticationRequiredError as e:
            logger.warning(f"Authentication required for MCP endpoint {mcp_endpoint.uri}: {e}")
            logger.info(
                f"Returning empty tool list for toolgroup {tool_group_id} - tools will be refreshed when authentication is available"
            )
            # Return empty list on authentication errors during startup
            # Tools will be refreshed when a turn is created with proper auth
            return ListToolDefsResponse(data=[])
        except Exception as e:
            logger.error(f"Failed to list tools for toolgroup {tool_group_id} at endpoint {mcp_endpoint.uri}: {e}")
            # Return empty list on other errors too to prevent crashes
            return ListToolDefsResponse(data=[])

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        if self.tool_store is None:
            raise ValueError(f"Tool store is not available for tool {tool_name}")

        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint = tool.metadata.get("endpoint")
        if endpoint is None:
            raise ValueError(f"Tool {tool_name} does not have an endpoint")
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        headers = await self.get_headers_from_request(endpoint)
        return await invoke_mcp_tool(endpoint, headers, tool_name, kwargs)

    async def get_headers_from_request(self, mcp_endpoint_uri: str) -> dict[str, str]:
        def canonicalize_uri(uri: str) -> str:
            return f"{urlparse(uri).netloc or ''}/{urlparse(uri).path or ''}"

        headers = {}

        provider_data = self.get_request_provider_data()
        if provider_data and provider_data.mcp_headers:
            for uri, values in provider_data.mcp_headers.items():
                if canonicalize_uri(uri) != canonicalize_uri(mcp_endpoint_uri):
                    continue
                headers.update(values)
        return headers
