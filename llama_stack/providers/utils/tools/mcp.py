# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import httpx
from mcp import ClientSession, McpError
from mcp import types as mcp_types
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from llama_stack.apis.common.content_types import ImageContentItem, InterleavedContentItem, TextContentItem
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
)
from llama_stack.distribution.datatypes import AuthenticationRequiredError
from llama_stack.log import get_logger
from llama_stack.providers.utils.tools.ttl_dict import TTLDict

logger = get_logger(__name__, category="tools")

protocol_cache = TTLDict(ttl_seconds=3600)

PROTOCOL_STREAMABLEHTTP = "streamable_http"
PROTOCOL_SSE = "sse"
PROTOCOL_UNKNOWN = "unknown"


@asynccontextmanager
async def client_wrapper(endpoint: str, headers: dict[str, str]) -> AsyncGenerator[ClientSession, Any]:
    # we use a ttl'd dict to cache the happy path protocol for each endpoint
    # but, we always fall back to trying the other protocol if we cannot initialize the session
    connection_strategies = [PROTOCOL_STREAMABLEHTTP, PROTOCOL_SSE]
    mcp_protocol = protocol_cache.get(endpoint, default=PROTOCOL_UNKNOWN)
    if mcp_protocol == PROTOCOL_SSE:
        connection_strategies = [PROTOCOL_SSE, PROTOCOL_STREAMABLEHTTP]

    for i, strategy in enumerate(connection_strategies):
        try:
            if strategy == PROTOCOL_STREAMABLEHTTP:
                async with streamablehttp_client(endpoint, headers=headers) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        protocol_cache[endpoint] = PROTOCOL_STREAMABLEHTTP
                        yield session
                        return
            elif strategy == PROTOCOL_SSE:
                async with sse_client(endpoint, headers=headers) as streams:
                    async with ClientSession(*streams) as session:
                        await session.initialize()
                        protocol_cache[endpoint] = PROTOCOL_SSE
                        yield session
                        return
        except* httpx.HTTPStatusError as eg:
            for exc in eg.exceptions:
                # mypy does not currently narrow the type of `eg.exceptions` based on the `except*` filter,
                # so we explicitly cast each item to httpx.HTTPStatusError. This is safe because
                # `except* httpx.HTTPStatusError` guarantees all exceptions in `eg.exceptions` are of that type.
                err = cast(httpx.HTTPStatusError, exc)
                if err.response.status_code == 401:
                    raise AuthenticationRequiredError(exc) from exc
            if i == len(connection_strategies) - 1:
                raise
        except* McpError:
            if i < len(connection_strategies) - 1:
                logger.warning(f"failed to connect via {strategy}, falling back to {connection_strategies[i + 1]}")
            else:
                raise


async def list_mcp_tools(endpoint: str, headers: dict[str, str]) -> ListToolDefsResponse:
    tools = []
    async with client_wrapper(endpoint, headers) as session:
        logger.debug("listing mcp tools")
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
                        "endpoint": endpoint,
                    },
                )
            )
    return ListToolDefsResponse(data=tools)


async def invoke_mcp_tool(
    endpoint: str, headers: dict[str, str], tool_name: str, kwargs: dict[str, Any]
) -> ToolInvocationResult:
    async with client_wrapper(endpoint, headers) as session:
        logger.debug("invoking mcp tool")
        result = await session.call_tool(tool_name, kwargs)

        content: list[InterleavedContentItem] = []
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
