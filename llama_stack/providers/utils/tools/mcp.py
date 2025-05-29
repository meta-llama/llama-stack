# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from contextlib import asynccontextmanager
from typing import Any

try:
    # for python < 3.11
    import exceptiongroup

    BaseExceptionGroup = exceptiongroup.BaseExceptionGroup
except ImportError:
    pass

import httpx
from mcp import ClientSession
from mcp import types as mcp_types
from mcp.client.sse import sse_client

from llama_stack.apis.common.content_types import ImageContentItem, InterleavedContentItem, TextContentItem
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
)
from llama_stack.distribution.datatypes import AuthenticationRequiredError
from llama_stack.log import get_logger

logger = get_logger(__name__, category="tools")


@asynccontextmanager
async def sse_client_wrapper(endpoint: str, headers: dict[str, str]):
    try:
        async with sse_client(endpoint, headers=headers) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                yield session
    except BaseException as e:
        if isinstance(e, BaseExceptionGroup):
            for exc in e.exceptions:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 401:
                    raise AuthenticationRequiredError(exc) from exc
        elif isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 401:
            raise AuthenticationRequiredError(e) from e

        raise


async def list_mcp_tools(endpoint: str, headers: dict[str, str]) -> ListToolDefsResponse:
    tools = []
    async with sse_client_wrapper(endpoint, headers) as session:
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
    async with sse_client_wrapper(endpoint, headers) as session:
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
