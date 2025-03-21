# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
import re
from typing import Dict, List

from pydantic import BaseModel

from llama_stack.apis.tools import RAGToolRuntime, SpecialToolGroup
from llama_stack.apis.version import LLAMA_STACK_API_VERSION
from llama_stack.distribution.resolver import api_protocol_map
from llama_stack.providers.datatypes import Api


class ApiEndpoint(BaseModel):
    route: str
    method: str
    name: str
    descriptive_name: str | None = None


def toolgroup_protocol_map():
    return {
        SpecialToolGroup.rag_tool: RAGToolRuntime,
    }


def get_all_api_endpoints() -> Dict[Api, List[ApiEndpoint]]:
    apis = {}

    protocols = api_protocol_map()
    toolgroup_protocols = toolgroup_protocol_map()
    for api, protocol in protocols.items():
        endpoints = []
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

            webmethod = method.__webmethod__
            route = f"/{LLAMA_STACK_API_VERSION}/{webmethod.route.lstrip('/')}"
            if webmethod.method == "GET":
                method = "get"
            elif webmethod.method == "DELETE":
                method = "delete"
            else:
                method = "post"
            endpoints.append(
                ApiEndpoint(route=route, method=method, name=name, descriptive_name=webmethod.descriptive_name)
            )

        apis[api] = endpoints

    return apis


def initialize_endpoint_impls(impls):
    endpoints = get_all_api_endpoints()
    endpoint_impls = {}

    def _convert_path_to_regex(path: str) -> str:
        # Convert {param} to named capture groups
        # handle {param:path} as well which allows for forward slashes in the param value
        pattern = re.sub(
            r"{(\w+)(?::path)?}",
            lambda m: f"(?P<{m.group(1)}>{'[^/]+' if not m.group(0).endswith(':path') else '.+'})",
            path,
        )

        return f"^{pattern}$"

    for api, api_endpoints in endpoints.items():
        if api not in impls:
            continue
        for endpoint in api_endpoints:
            impl = impls[api]
            func = getattr(impl, endpoint.name)
            if endpoint.method not in endpoint_impls:
                endpoint_impls[endpoint.method] = {}
            endpoint_impls[endpoint.method][_convert_path_to_regex(endpoint.route)] = (
                func,
                endpoint.descriptive_name or endpoint.route,
            )

    return endpoint_impls


def find_matching_endpoint(method, path, endpoint_impls):
    """Find the matching endpoint implementation for a given method and path.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: URL path to match against
        endpoint_impls: A dictionary of endpoint implementations

    Returns:
        A tuple of (endpoint_function, path_params, descriptive_name)

    Raises:
        ValueError: If no matching endpoint is found
    """
    impls = endpoint_impls.get(method.lower())
    if not impls:
        raise ValueError(f"No endpoint found for {path}")

    for regex, (func, descriptive_name) in impls.items():
        match = re.match(regex, path)
        if match:
            # Extract named groups from the regex match
            path_params = match.groupdict()
            return func, path_params, descriptive_name

    raise ValueError(f"No endpoint found for {path}")
