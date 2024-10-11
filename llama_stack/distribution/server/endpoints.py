# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
from typing import Dict, List

from pydantic import BaseModel

from llama_stack.distribution.resolver import api_protocol_map

from llama_stack.providers.datatypes import Api


class ApiEndpoint(BaseModel):
    route: str
    method: str
    name: str


def get_all_api_endpoints() -> Dict[Api, List[ApiEndpoint]]:
    apis = {}

    protocols = api_protocol_map()
    for api, protocol in protocols.items():
        endpoints = []
        protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)

        for name, method in protocol_methods:
            if not hasattr(method, "__webmethod__"):
                continue

            webmethod = method.__webmethod__
            route = webmethod.route

            if webmethod.method == "GET":
                method = "get"
            elif webmethod.method == "DELETE":
                method = "delete"
            else:
                method = "post"
            endpoints.append(ApiEndpoint(route=route, method=method, name=name))

        apis[api] = endpoints

    return apis
