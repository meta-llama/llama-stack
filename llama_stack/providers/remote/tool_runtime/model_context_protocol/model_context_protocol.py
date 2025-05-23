# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

import exceptiongroup
import httpx
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.sse import sse_client

from llama_stack.apis.common.content_types import URL, ImageContentItem, TextContentItem
from llama_stack.apis.datatypes import Api
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.distribution.datatypes import AuthenticationRequiredError
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from .config import MCPProviderConfig

logger = get_logger(__name__, category="tools")


@asynccontextmanager
async def sse_client_wrapper(endpoint: str, headers: dict[str, str]):
    try:
        async with sse_client(endpoint, headers=headers) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                yield session
    except BaseException as e:
        if isinstance(e, exceptiongroup.BaseExceptionGroup):
            for exc in e.exceptions:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 401:
                    raise AuthenticationRequiredError(exc) from exc
        elif isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 401:
            raise AuthenticationRequiredError(e) from e

        raise


class ModelContextProtocolToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    def __init__(self, config: MCPProviderConfig, _deps: dict[Api, Any]):
        self.config = config

    async def initialize(self):
        pass

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        # this endpoint should be retrieved by getting the tool group right?
        if mcp_endpoint is None:
            raise ValueError("mcp_endpoint is required")

        headers = await self.get_headers_from_request(mcp_endpoint.uri)
        tools = []
        async with sse_client_wrapper(mcp_endpoint.uri, headers) as session:
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                parameters = []
                for param_name, param_schema in tool.inputSchema.get("properties", {}).items():
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
                            "endpoint": mcp_endpoint.uri,
                        },
                    )
                )
        return ListToolDefsResponse(data=tools)

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        tool = await self.tool_store.get_tool(tool_name)
        if tool.metadata is None or tool.metadata.get("endpoint") is None:
            raise ValueError(f"Tool {tool_name} does not have metadata")
        endpoint = tool.metadata.get("endpoint")
        if urlparse(endpoint).scheme not in ("http", "https"):
            raise ValueError(f"Endpoint {endpoint} is not a valid HTTP(S) URL")

        headers = await self.get_headers_from_request(endpoint)
        async with sse_client_wrapper(endpoint, headers) as session:
            result = await session.call_tool(tool.identifier, kwargs)

        content = []
        for item in result.content:
            if isinstance(item, mcp_types.TextContent):
                content.append(TextContentItem(text=item.text))
            elif isinstance(item, mcp_types.ImageContent):
                content.append(ImageContentItem(image=item.data))
            elif isinstance(item, mcp_types.EmbeddedResource):
                logger.warning(f"EmbeddedResource is not supported: {item}")
            else:
                raise ValueError(f"Unknown content type: {type(item)}")
        return ToolInvocationResult(
            content=content,
            error_code=1 if result.isError else 0,
        )

    async def get_headers_from_request(self, mcp_endpoint_uri: str) -> dict[str, str]:
        def canonicalize_uri(uri: str) -> str:
            return f"{urlparse(uri).netloc or ''}/{urlparse(uri).path or ''}"

        headers = {}

        provider_data = self.get_request_provider_data()
        if provider_data and provider_data.mcp_headers:
            for uri, values in provider_data.mcp_headers.items():
                if canonicalize_uri(uri) != canonicalize_uri(mcp_endpoint_uri):
                    continue
                for entry in values:
                    parts = entry.split(":")
                    if len(parts) == 2:
                        k, v = parts
                        headers[k.strip()] = v.strip()
        return headers
