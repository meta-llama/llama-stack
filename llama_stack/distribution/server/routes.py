# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import re
from collections.abc import Callable
from typing import Any

from aiohttp import hdrs
from starlette.routing import Route

from llama_stack.apis.tools import RAGToolRuntime, SpecialToolGroup
from llama_stack.apis.version import LLAMA_STACK_API_VERSION
from llama_stack.distribution.resolver import api_protocol_map
from llama_stack.providers.datatypes import Api

EndpointFunc = Callable[..., Any]
PathParams = dict[str, str]
RouteInfo = tuple[EndpointFunc, str]
PathImpl = dict[str, RouteInfo]
RouteImpls = dict[str, PathImpl]
RouteMatch = tuple[EndpointFunc, PathParams, str]


def toolgroup_protocol_map():
    return {
        SpecialToolGroup.rag_tool: RAGToolRuntime,
    }


def get_all_api_routes() -> dict[Api, list[Route]]:
    apis = {}

    protocols = api_protocol_map()
    toolgroup_protocols = toolgroup_protocol_map()
    for api, protocol in protocols.items():
        routes = []
        protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)

        # HACK ALERT
        if api == Api.tool_runtime:
            for tool_group in SpecialToolGroup:
                sub_protocol = toolgroup_protocols[tool_group]
                sub_protocol_methods = inspect.getmembers(sub_protocol, predicate=inspect.isfunction)
                for name, method in sub_protocol_methods:
                    if not hasattr(method, "__webmethod__"):
                        continue
                    protocol_methods.append((f"{tool_group.value}.{name}", method))

        for name, method in protocol_methods:
            if not hasattr(method, "__webmethod__"):
                continue

            # The __webmethod__ attribute is dynamically added by the @webmethod decorator
            # mypy doesn't know about this dynamic attribute, so we ignore the attr-defined error
            webmethod = method.__webmethod__  # type: ignore[attr-defined]
            path = f"/{LLAMA_STACK_API_VERSION}/{webmethod.route.lstrip('/')}"
            if webmethod.method == hdrs.METH_GET:
                http_method = hdrs.METH_GET
            elif webmethod.method == hdrs.METH_DELETE:
                http_method = hdrs.METH_DELETE
            elif webmethod.method == hdrs.METH_PUT:
                http_method = hdrs.METH_PUT
            else:
                http_method = hdrs.METH_POST
            routes.append(
                Route(path=path, methods=[http_method], name=name, endpoint=None)
            )  # setting endpoint to None since don't use a Router object

        apis[api] = routes

    return apis


def initialize_route_impls(impls: dict[Api, Any]) -> RouteImpls:
    routes = get_all_api_routes()
    route_impls: RouteImpls = {}

    def _convert_path_to_regex(path: str) -> str:
        # Convert {param} to named capture groups
        # handle {param:path} as well which allows for forward slashes in the param value
        pattern = re.sub(
            r"{(\w+)(?::path)?}",
            lambda m: f"(?P<{m.group(1)}>{'[^/]+' if not m.group(0).endswith(':path') else '.+'})",
            path,
        )

        return f"^{pattern}$"

    for api, api_routes in routes.items():
        if api not in impls:
            continue
        for route in api_routes:
            impl = impls[api]
            func = getattr(impl, route.name)
            # Get the first (and typically only) method from the set, filtering out HEAD
            available_methods = [m for m in route.methods if m != "HEAD"]
            if not available_methods:
                continue  # Skip if only HEAD method is available
            method = available_methods[0].lower()
            if method not in route_impls:
                route_impls[method] = {}
            route_impls[method][_convert_path_to_regex(route.path)] = (
                func,
                route.path,
            )

    return route_impls


def find_matching_route(method: str, path: str, route_impls: RouteImpls) -> RouteMatch:
    """Find the matching endpoint implementation for a given method and path.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: URL path to match against
        route_impls: A dictionary of endpoint implementations

    Returns:
        A tuple of (endpoint_function, path_params, descriptive_name)

    Raises:
        ValueError: If no matching endpoint is found
    """
    impls = route_impls.get(method.lower())
    if not impls:
        raise ValueError(f"No endpoint found for {path}")

    for regex, (func, descriptive_name) in impls.items():
        match = re.match(regex, path)
        if match:
            # Extract named groups from the regex match
            path_params = match.groupdict()
            return func, path_params, descriptive_name

    raise ValueError(f"No endpoint found for {path}")
