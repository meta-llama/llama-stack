# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import re
import tempfile

from abc import abstractmethod
from typing import List, Optional

import requests
from termcolor import cprint

from .ipython_tool.code_execution import (
    CodeExecutionContext,
    CodeExecutionRequest,
    CodeExecutor,
    TOOLS_ATTACHMENT_KEY_REGEX,
)

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.agents import *  # noqa: F403

from .base import BaseTool


def interpret_content_as_attachment(content: str) -> Optional[Attachment]:
    match = re.search(TOOLS_ATTACHMENT_KEY_REGEX, content)
    if match:
        snippet = match.group(1)
        data = json.loads(snippet)
        return Attachment(
            content=URL(uri="file://" + data["filepath"]), mime_type=data["mimetype"]
        )

    return None


class SingleMessageBuiltinTool(BaseTool):
    async def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        assert len(messages) == 1, f"Expected single message, got {len(messages)}"

        message = messages[0]
        assert len(message.tool_calls) == 1, "Expected a single tool call"

        tool_call = messages[0].tool_calls[0]

        query = tool_call.arguments["query"]
        response: str = await self.run_impl(query)

        message = ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=response,
        )
        return [message]

    @abstractmethod
    async def run_impl(self, query: str) -> str:
        raise NotImplementedError()


class PhotogenTool(SingleMessageBuiltinTool):
    def __init__(self, dump_dir: str) -> None:
        self.dump_dir = dump_dir

    def get_name(self) -> str:
        return BuiltinTool.photogen.value

    async def run_impl(self, query: str) -> str:
        """
        Implement this to give the model an ability to generate images.

        Return:
            info = {
                "filepath": str(image_filepath),
                "mimetype": "image/png",
            }
        """
        raise NotImplementedError()


class SearchTool(SingleMessageBuiltinTool):
    def __init__(self, engine: SearchEngineType, api_key: str, **kwargs) -> None:
        self.api_key = api_key
        if engine == SearchEngineType.bing:
            self.engine = BingSearch(api_key, **kwargs)
        elif engine == SearchEngineType.brave:
            self.engine = BraveSearch(api_key, **kwargs)
        else:
            raise ValueError(f"Unknown search engine: {engine}")

    def get_name(self) -> str:
        return BuiltinTool.brave_search.value

    async def run_impl(self, query: str) -> str:
        return await self.engine.search(query)


class BingSearch:
    def __init__(self, api_key: str, top_k: int = 3, **kwargs) -> None:
        self.api_key = api_key
        self.top_k = top_k

    async def search(self, query: str) -> str:
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
        }
        params = {
            "count": self.top_k,
            "textDecorations": True,
            "textFormat": "HTML",
            "q": query,
        }

        response = requests.get(url=url, params=params, headers=headers)
        response.raise_for_status()
        clean = self._clean_response(response.json())
        return json.dumps(clean)

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

        return {"query": query, "top_k": clean_response}


class BraveSearch:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def search(self, query: str) -> str:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept-Encoding": "gzip",
            "Accept": "application/json",
        }
        payload = {"q": query}
        response = requests.get(url=url, params=payload, headers=headers)
        return json.dumps(self._clean_brave_response(response.json()))

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
                if r_type == "web":
                    # For web data - add a single output from the search
                    idx = m["index"]
                    selected_keys = [
                        "type",
                        "title",
                        "url",
                        "description",
                        "date",
                        "extra_snippets",
                    ]
                    cleaned = {
                        k: v for k, v in results[idx].items() if k in selected_keys
                    }
                elif r_type == "faq":
                    # For faw data - take a list of all the questions & answers
                    selected_keys = ["type", "question", "answer", "title", "url"]
                    cleaned = []
                    for q in results:
                        cleaned.append(
                            {k: v for k, v in q.items() if k in selected_keys}
                        )
                elif r_type == "infobox":
                    idx = m["index"]
                    selected_keys = [
                        "type",
                        "title",
                        "url",
                        "description",
                        "long_desc",
                    ]
                    cleaned = {
                        k: v for k, v in results[idx].items() if k in selected_keys
                    }
                elif r_type == "videos":
                    selected_keys = [
                        "type",
                        "url",
                        "title",
                        "description",
                        "date",
                    ]
                    cleaned = []
                    for q in results:
                        cleaned.append(
                            {k: v for k, v in q.items() if k in selected_keys}
                        )
                elif r_type == "locations":
                    # For faw data - take a list of all the questions & answers
                    selected_keys = [
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
                    ]
                    cleaned = []
                    for q in results:
                        cleaned.append(
                            {k: v for k, v in q.items() if k in selected_keys}
                        )
                elif r_type == "news":
                    # For faw data - take a list of all the questions & answers
                    selected_keys = [
                        "type",
                        "title",
                        "url",
                        "description",
                    ]
                    cleaned = []
                    for q in results:
                        cleaned.append(
                            {k: v for k, v in q.items() if k in selected_keys}
                        )
                else:
                    cleaned = []

                clean_response.append(cleaned)

        return {"query": query, "top_k": clean_response}


class WolframAlphaTool(SingleMessageBuiltinTool):
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.url = "https://api.wolframalpha.com/v2/query"

    def get_name(self) -> str:
        return BuiltinTool.wolfram_alpha.value

    async def run_impl(self, query: str) -> str:
        params = {
            "input": query,
            "appid": self.api_key,
            "format": "plaintext",
            "output": "json",
        }
        response = requests.get(
            self.url,
            params=params,
        )

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
        for main_key in remove:
            for key_to_remove in remove[main_key]:
                try:
                    if key_to_remove == "assumptions":
                        if "assumptions" in wa_response[main_key]:
                            del wa_response[main_key][key_to_remove]
                    if isinstance(key_to_remove, dict):
                        for sub_key in key_to_remove:
                            if sub_key == "pods":
                                for i in range(len(wa_response[main_key][sub_key])):
                                    if (
                                        wa_response[main_key][sub_key][i]["title"]
                                        == "Result"
                                    ):
                                        del wa_response[main_key][sub_key][i + 1 :]
                                        break
                            sub_items = wa_response[main_key][sub_key]
                            for i in range(len(sub_items)):
                                for sub_key_to_remove in key_to_remove[sub_key]:
                                    if sub_key_to_remove in sub_items[i]:
                                        del sub_items[i][sub_key_to_remove]
                    elif key_to_remove in wa_response[main_key]:
                        del wa_response[main_key][key_to_remove]
                except KeyError:
                    pass
        return wa_response


class CodeInterpreterTool(BaseTool):
    def __init__(self) -> None:
        ctx = CodeExecutionContext(
            matplotlib_dump_dir=tempfile.mkdtemp(),
        )
        self.code_executor = CodeExecutor(ctx)

    def get_name(self) -> str:
        return BuiltinTool.code_interpreter.value

    async def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        message = messages[0]
        assert len(message.tool_calls) == 1, "Expected a single tool call"

        tool_call = messages[0].tool_calls[0]
        script = tool_call.arguments["code"]

        req = CodeExecutionRequest(scripts=[script])
        res = self.code_executor.execute(req)

        pieces = [res["process_status"]]
        for out_type in ["stdout", "stderr"]:
            res_out = res[out_type]
            if res_out != "":
                pieces.extend([f"[{out_type}]", res_out, f"[/{out_type}]"])
                if out_type == "stderr":
                    cprint(f"ipython tool error: ↓\n{res_out}", color="red")

        message = ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content="\n".join(pieces),
        )
        return [message]
