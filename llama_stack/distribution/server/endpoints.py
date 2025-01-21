# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
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
                sub_protocol_methods = inspect.getmembers(
                    sub_protocol, predicate=inspect.isfunction
                )
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
            endpoints.append(ApiEndpoint(route=route, method=method, name=name))

        apis[api] = endpoints

    return apis
