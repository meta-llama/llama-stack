# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
from typing import Dict, List

from pydantic import BaseModel

from llama_stack.apis.agents import Agents
from llama_stack.apis.inference import Inference
from llama_stack.apis.inspect import Inspect
from llama_stack.apis.memory import Memory
from llama_stack.apis.memory_banks import MemoryBanks
from llama_stack.apis.models import Models
from llama_stack.apis.safety import Safety
from llama_stack.apis.shields import Shields
from llama_stack.apis.telemetry import Telemetry

from llama_stack.providers.datatypes import Api


class ApiEndpoint(BaseModel):
    route: str
    method: str
    name: str


def get_all_api_endpoints() -> Dict[Api, List[ApiEndpoint]]:
    apis = {}

    protocols = {
        Api.inference: Inference,
        Api.safety: Safety,
        Api.agents: Agents,
        Api.memory: Memory,
        Api.telemetry: Telemetry,
        Api.models: Models,
        Api.shields: Shields,
        Api.memory_banks: MemoryBanks,
        Api.inspect: Inspect,
    }

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
