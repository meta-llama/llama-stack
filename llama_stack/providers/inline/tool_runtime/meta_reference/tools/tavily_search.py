# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List

import requests

from llama_stack.providers.inline.tool_runtime.meta_reference.tools.base import BaseTool
from pydantic import BaseModel


class TavilySearchConfig(BaseModel):
    api_key: str
    max_results: int = 3


class TavilySearchTool(BaseTool):
    requires_api_key: bool = True

    @classmethod
    def tool_id(cls) -> str:
        return "tavily_search"

    @classmethod
    def get_provider_config_type(cls):
        return TavilySearchConfig

    async def execute(self, query: str) -> List[dict]:
        config = TavilySearchConfig(**self.config)
        response = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": config.api_key, "query": query},
        )
        response.raise_for_status()
        search_response = response.json()
        return {
            "query": search_response["query"],
            "results": search_response["results"][: config.max_results],
        }
