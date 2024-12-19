# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List

import requests

from llama_stack.apis.tools import InvokeToolResult, Tool, ToolRuntime
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from .config import BraveSearchToolConfig


class BraveSearchToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime):
    def __init__(self, config: BraveSearchToolConfig):
        self.config = config

    async def initialize(self):
        pass

    async def register_tool(self, tool: Tool):
        if tool.identifier != "brave_search":
            raise ValueError(f"Tool identifier {tool.identifier} is not supported")

    async def unregister_tool(self, tool_id: str) -> None:
        return

    async def invoke_tool(self, tool_id: str, args: Dict[str, Any]) -> InvokeToolResult:
        results = await self.execute(args["query"])
        content_items = "\n".join([str(result) for result in results])
        return InvokeToolResult(
            content=content_items,
        )

    async def execute(self, query: str) -> List[dict]:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": self.config.api_key,
            "Accept-Encoding": "gzip",
            "Accept": "application/json",
        }
        payload = {"q": query}
        response = requests.get(url=url, params=payload, headers=headers)
        response.raise_for_status()
        return self._clean_brave_response(response.json())

    def _clean_brave_response(self, search_response):
        clean_response = []
        if "mixed" in search_response:
            mixed_results = search_response["mixed"]
            for m in mixed_results["main"][: self.config.max_results]:
                r_type = m["type"]
                results = search_response[r_type]["results"]
                cleaned = self._clean_result_by_type(r_type, results, m.get("index"))
                clean_response.append(cleaned)

        return clean_response

    def _clean_result_by_type(self, r_type, results, idx=None):
        type_cleaners = {
            "web": (
                ["type", "title", "url", "description", "date", "extra_snippets"],
                lambda x: x[idx],
            ),
            "faq": (["type", "question", "answer", "title", "url"], lambda x: x),
            "infobox": (
                ["type", "title", "url", "description", "long_desc"],
                lambda x: x[idx],
            ),
            "videos": (["type", "url", "title", "description", "date"], lambda x: x),
            "locations": (
                [
                    "type",
                    "title",
                    "url",
                    "description",
                    "coordinates",
                    "postal_address",
                    "contact",
                    "rating",
                    "distance",
                    "zoom_level",
                ],
                lambda x: x,
            ),
            "news": (["type", "title", "url", "description"], lambda x: x),
        }

        if r_type not in type_cleaners:
            return ""

        selected_keys, result_selector = type_cleaners[r_type]
        results = result_selector(results)

        if isinstance(results, list):
            cleaned = [
                {k: v for k, v in item.items() if k in selected_keys}
                for item in results
            ]
        else:
            cleaned = {k: v for k, v in results.items() if k in selected_keys}

        return str(cleaned)
