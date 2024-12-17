# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Dict

import requests

from llama_stack.providers.inline.tool_runtime.meta_reference.tools.base import BaseTool
from pydantic import BaseModel


class WolframAlphaConfig(BaseModel):
    api_key: str


class WolframAlphaTool(BaseTool):
    requires_api_key: bool = True

    @classmethod
    def tool_id(cls) -> str:
        return "wolfram_alpha"

    @classmethod
    def get_provider_config_type(cls):
        return WolframAlphaConfig

    async def execute(self, query: str) -> Dict:
        config = WolframAlphaConfig(**self.config)
        url = "https://api.wolframalpha.com/v2/query"
        params = {
            "input": query,
            "appid": config.api_key,
            "format": "plaintext",
            "output": "json",
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return json.dumps(self._clean_wolfram_alpha_response(response.json()))

    def _clean_wolfram_alpha_response(self, wa_response):
        remove = {
            "queryresult": [
                "datatypes",
                "error",
                "timedout",
                "timedoutpods",
                "numpods",
                "timing",
                "parsetiming",
                "parsetimedout",
                "recalculate",
                "id",
                "host",
                "server",
                "related",
                "version",
                {
                    "pods": [
                        "scanner",
                        "id",
                        "error",
                        "expressiontypes",
                        "states",
                        "infos",
                        "position",
                        "numsubpods",
                    ]
                },
                "assumptions",
            ],
        }

        result = wa_response.copy()
        for main_key, to_remove in remove.items():
            if main_key not in result:
                continue

            for item in to_remove:
                if isinstance(item, dict):
                    for sub_key, sub_items in item.items():
                        if sub_key == "pods":
                            pods = result[main_key].get(sub_key, [])
                            for i, pod in enumerate(pods):
                                if pod.get("title") == "Result":
                                    pods = pods[: i + 1]
                                    break
                                for remove_key in sub_items:
                                    pod.pop(remove_key, None)
                else:
                    result[main_key].pop(item, None)

        return result
