# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging

import requests

logger = logging.getLogger(__name__)


async def bing_search(query: str, __api_key__: str, top_k: int = 3, **kwargs) -> str:
    url = "https://api.bing.microsoft.com/v7.0/search"
    headers = {
        "Ocp-Apim-Subscription-Key": __api_key__,
    }
    params = {
        "count": top_k,
        "textDecorations": True,
        "textFormat": "HTML",
        "q": query,
    }

    response = requests.get(url=url, params=params, headers=headers)
    response.raise_for_status()
    clean = _bing_clean_response(response.json())
    return json.dumps(clean)


def _bing_clean_response(search_response):
    clean_response = []
    query = search_response["queryContext"]["originalQuery"]
    if "webPages" in search_response:
        pages = search_response["webPages"]["value"]
        for p in pages:
            selected_keys = {"name", "url", "snippet"}
            clean_response.append({k: v for k, v in p.items() if k in selected_keys})
    if "news" in search_response:
        clean_news = []
        news = search_response["news"]["value"]
        for n in news:
            selected_keys = {"name", "url", "description"}
            clean_news.append({k: v for k, v in n.items() if k in selected_keys})

        clean_response.append(clean_news)

    return {"query": query, "top_k": clean_response}


async def brave_search(query: str, __api_key__: str) -> str:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "X-Subscription-Token": __api_key__,
        "Accept-Encoding": "gzip",
        "Accept": "application/json",
    }
    payload = {"q": query}
    response = requests.get(url=url, params=payload, headers=headers)
    return json.dumps(_clean_brave_response(response.json()))


def _clean_brave_response(search_response, top_k=3):
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
                cleaned = {k: v for k, v in results[idx].items() if k in selected_keys}
            elif r_type == "faq":
                # For faw data - take a list of all the questions & answers
                selected_keys = ["type", "question", "answer", "title", "url"]
                cleaned = []
                for q in results:
                    cleaned.append({k: v for k, v in q.items() if k in selected_keys})
            elif r_type == "infobox":
                idx = m["index"]
                selected_keys = [
                    "type",
                    "title",
                    "url",
                    "description",
                    "long_desc",
                ]
                cleaned = {k: v for k, v in results[idx].items() if k in selected_keys}
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
                    cleaned.append({k: v for k, v in q.items() if k in selected_keys})
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
                    cleaned.append({k: v for k, v in q.items() if k in selected_keys})
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
                    cleaned.append({k: v for k, v in q.items() if k in selected_keys})
            else:
                cleaned = []

            clean_response.append(cleaned)

    return {"query": query, "top_k": clean_response}


async def tavily_search(query: str, __api_key__: str) -> str:
    response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": __api_key__, "query": query},
    )
    return json.dumps(_clean_tavily_response(response.json()))


def _clean_tavily_response(search_response, top_k=3):
    return {"query": search_response["query"], "top_k": search_response["results"]}


async def print_tool(query: str, __api_key__: str) -> str:
    logger.info(f"print_tool called with query: {query} and api_key: {__api_key__}")
    return json.dumps({"result": "success"})
