# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

"""Shared shape for web_search tool_runtime providers' result metadata.

Each web_search provider (bing, brave, exa, nimble, serply, tavily) returns
ToolInvocationResult.metadata["sources"]: a list of source dicts, read independently by the
Responses layer to build both web_search_call.action.sources and url_citation annotations
(see tool_executor.py). Since metadata is an untyped dict[str, Any], a shared TypedDict and
parser keep providers and both readers from drifting apart on the shape or on how a malformed
entry gets handled.
"""

from typing import Any, TypedDict

from ogx.log import get_logger

logger = get_logger(name=__name__, category="tool_runtime")


class WebSearchSourceDict(TypedDict):
    """One web_search result source, as providers should build it for
    ToolInvocationResult.metadata["sources"]."""

    url: str
    title: str


def parse_web_search_sources(raw_sources: Any) -> list[WebSearchSourceDict]:
    """Validate and parse a raw metadata["sources"] value into well-formed WebSearchSourceDicts.

    Providers are external services; a malformed entry (metadata["sources"] itself not a
    list, an element that isn't a dict, or a non-string/empty url) is skipped and logged, not
    raised, since raising here would otherwise fail a web_search call that already succeeded
    (see tool_executor.py, which builds citations from this after the tool call completed).
    A missing or non-string title falls back to the url.
    """
    if not isinstance(raw_sources, list):
        if raw_sources:
            logger.warning("Ignoring non-list web_search sources metadata", sources=repr(raw_sources))
        return []

    parsed: list[WebSearchSourceDict] = []
    for source in raw_sources:
        if not isinstance(source, dict):
            logger.warning("Skipping malformed web_search source: not a dict", source=repr(source))
            continue
        url = source.get("url")
        if not isinstance(url, str) or not url:
            logger.warning("Skipping web_search source with a missing or non-string url", source=repr(source))
            continue
        title = source.get("title")
        if not isinstance(title, str) or not title:
            title = url
        parsed.append(WebSearchSourceDict(url=url, title=title))
    return parsed
