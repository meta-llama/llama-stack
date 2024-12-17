# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

import requests

from llama_stack.providers.inline.tool_runtime.meta_reference.tools.base import BaseTool
from pydantic import BaseModel


class BraveSearchConfig(BaseModel):
    api_key: str
    max_results: int = 3


class BraveSearchTool(BaseTool):
    requires_api_key: bool = True

    @classmethod
    def tool_id(cls) -> str:
        return "brave_search"

    @classmethod
    def get_provider_config_type(cls):
        return BraveSearchConfig

    async def execute(self, query: str) -> List[dict]:
        config = BraveSearchConfig(**self.config)
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": config.api_key,
            "Accept-Encoding": "gzip",
            "Accept": "application/json",
        }
        payload = {"q": query}
        response = requests.get(url=url, params=payload, headers=headers)
        response.raise_for_status()
        return self._clean_brave_response(response.json(), config.max_results)

    def _clean_brave_response(self, search_response, top_k=3):
        query = None
        clean_response = []
        if "query" in search_response:
            if "original" in search_response["query"]:
                query = search_response["query"]["original"]
        if "mixed" in search_response:
            mixed_results = search_response["mixed"]
            for m in mixed_results["main"][:top_k]:
                r_type = m["type"]
                results = search_response[r_type]["results"]
                cleaned = self._clean_result_by_type(r_type, results, m.get("index"))
                clean_response.append(cleaned)

        return {"query": query, "results": clean_response}

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
            return []

        selected_keys, result_selector = type_cleaners[r_type]
        results = result_selector(results)

        if isinstance(results, list):
            return [
                {k: v for k, v in item.items() if k in selected_keys}
                for item in results
            ]
        return {k: v for k, v in results.items() if k in selected_keys}
