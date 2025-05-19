# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncGenerator
from typing import Any, cast
from urllib.parse import urlparse

import exceptiongroup
import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.datatypes import Api
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolDef,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.distribution.credentials import AuthenticationRequiredError, CredentialsStore
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from .config import ModelContextProtocolConfig


async def sse_client_wrapper(endpoint: str, headers: dict[str, str]) -> AsyncGenerator[ClientSession, None]:
    try:
        async with sse_client(endpoint, headers=headers) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                yield session
    except BaseException as e:
        # TODO: auto-discover auth metadata and cache it, add a nonce, create state
        # which can be used to exchange the authorization code for an access token.
        if isinstance(e, exceptiongroup.BaseExceptionGroup):
            for exc in e.exceptions:
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 401:
                    raise AuthenticationRequiredError(exc) from exc
        elif isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 401:
            raise AuthenticationRequiredError(e) from e
        else:
            raise


class ModelContextProtocolToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime):
    def __init__(self, config: ModelContextProtocolConfig, deps: dict[Api, Any]):
        self.config = config
        self.credentials_store = cast(CredentialsStore, deps[Api.credentials])

    async def initialize(self):
        pass

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        if mcp_endpoint is None:
            raise ValueError("mcp_endpoint is required")

        headers = await self.get_headers()
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

        headers = await self.get_headers()
        async with sse_client_wrapper(endpoint, headers) as session:
            result = await session.call_tool(tool.identifier, kwargs)

        return ToolInvocationResult(
            content=[result.model_dump_json() for result in result.content],
            error_code=1 if result.isError else 0,
        )

    async def get_headers(self) -> dict[str, str]:
        headers = {}
        credentials = await self.credentials_store.get_credential(self.__provider_id__)
        if credentials:
            headers["Authorization"] = f"Bearer {credentials.token}"
        return headers
