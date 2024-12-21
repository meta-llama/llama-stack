# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Any, Dict, List

import requests

from llama_stack.apis.tools import Tool, ToolGroupDef, ToolInvocationResult, ToolRuntime
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from .config import TavilySearchToolConfig


class TavilySearchToolRuntimeImpl(
    ToolsProtocolPrivate, ToolRuntime, NeedsRequestProviderData
):
    def __init__(self, config: TavilySearchToolConfig):
        self.config = config

    async def initialize(self):
        pass

    async def register_tool(self, tool: Tool):
        if tool.identifier != "tavily_search":
            raise ValueError(f"Tool identifier {tool.identifier} is not supported")

    async def unregister_tool(self, tool_id: str) -> None:
        return

    def _get_api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key

        provider_data = self.get_request_provider_data()
        if provider_data is None or not provider_data.api_key:
            raise ValueError(
                'Pass Search provider\'s API Key in the header X-LlamaStack-ProviderData as { "api_key": <your api key>}'
            )
        return provider_data.api_key

    async def discover_tools(self, tool_group: ToolGroupDef) -> List[Tool]:
        raise NotImplementedError("Tavily search tool group not supported")

    async def invoke_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolInvocationResult:
        api_key = self._get_api_key()
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": args["query"]},
        )

        return ToolInvocationResult(
            content=json.dumps(self._clean_tavily_response(response.json()))
        )

    def _clean_tavily_response(self, search_response, top_k=3):
        return {"query": search_response["query"], "top_k": search_response["results"]}
