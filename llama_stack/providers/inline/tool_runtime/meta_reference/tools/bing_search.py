# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import List

import requests

from llama_stack.providers.inline.tool_runtime.meta_reference.tools.base import BaseTool
from pydantic import BaseModel


class BingSearchConfig(BaseModel):
    api_key: str
    max_results: int = 5


class BingSearchTool(BaseTool):
    requires_api_key: bool = True

    @classmethod
    def tool_id(cls) -> str:
        return "bing_search"

    @classmethod
    def get_provider_config_type(cls):
        return BingSearchConfig

    async def execute(self, query: str) -> List[dict]:
        config = BingSearchConfig(**self.config)
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": config.api_key,
        }
        params = {
            "count": config.max_results,
            "textDecorations": True,
            "textFormat": "HTML",
            "q": query,
        }

        response = requests.get(url=url, params=params, headers=headers)
        response.raise_for_status()
        return json.dumps(self._clean_response(response.json()))

    def _clean_response(self, search_response):
        clean_response = []
        query = search_response["queryContext"]["originalQuery"]
        if "webPages" in search_response:
            pages = search_response["webPages"]["value"]
            for p in pages:
                selected_keys = {"name", "url", "snippet"}
                clean_response.append(
                    {k: v for k, v in p.items() if k in selected_keys}
                )
        if "news" in search_response:
            clean_news = []
            news = search_response["news"]["value"]
            for n in news:
                selected_keys = {"name", "url", "description"}
                clean_news.append({k: v for k, v in n.items() if k in selected_keys})
            clean_response.append(clean_news)

        return {"query": query, "results": clean_response}
