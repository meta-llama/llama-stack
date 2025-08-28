# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any

import httpx

from llama_stack.apis.common.content_types import URL
from llama_stack.apis.tools import (
    ListToolDefsResponse,
    ToolDef,
    ToolGroup,
    ToolInvocationResult,
    ToolParameter,
    ToolRuntime,
)
from llama_stack.core.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import ToolGroupsProtocolPrivate

from .config import TavilySearchToolConfig


class TavilySearchToolRuntimeImpl(ToolGroupsProtocolPrivate, ToolRuntime, NeedsRequestProviderData):
    def __init__(self, config: TavilySearchToolConfig):
        self.config = config

    async def initialize(self):
        pass

    async def register_toolgroup(self, toolgroup: ToolGroup) -> None:
        pass

    async def unregister_toolgroup(self, toolgroup_id: str) -> None:
        return

    def _get_api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key.get_secret_value()

        provider_data = self.get_request_provider_data()
        if provider_data is None or not provider_data.tavily_search_api_key:
            raise ValueError(
                'Pass Search provider\'s API Key in the header X-LlamaStack-Provider-Data as { "tavily_search_api_key": <your api key>}'
            )
        return provider_data.tavily_search_api_key

    async def list_runtime_tools(
        self, tool_group_id: str | None = None, mcp_endpoint: URL | None = None
    ) -> ListToolDefsResponse:
        return ListToolDefsResponse(
            data=[
                ToolDef(
                    name="web_search",
                    description="Search the web for information",
                    parameters=[
                        ToolParameter(
                            name="query",
                            description="The query to search for",
                            parameter_type="string",
                        )
                    ],
                )
            ]
        )

    async def invoke_tool(self, tool_name: str, kwargs: dict[str, Any]) -> ToolInvocationResult:
        api_key = self._get_api_key()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": kwargs["query"]},
            )
            response.raise_for_status()

        return ToolInvocationResult(content=json.dumps(self._clean_tavily_response(response.json())))

    def _clean_tavily_response(self, search_response, top_k=3):
        return {"query": search_response["query"], "top_k": search_response["results"]}
