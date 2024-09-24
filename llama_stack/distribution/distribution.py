# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
import inspect
from typing import Dict, List

from pydantic import BaseModel

from llama_stack.apis.agents import Agents
from llama_stack.apis.inference import Inference
from llama_stack.apis.memory import Memory
from llama_stack.apis.memory_banks import MemoryBanks
from llama_stack.apis.models import Models
from llama_stack.apis.safety import Safety
from llama_stack.apis.shields import Shields
from llama_stack.apis.telemetry import Telemetry

from .datatypes import Api, ApiEndpoint, ProviderSpec, remote_provider_spec

# These are the dependencies needed by the distribution server.
# `llama-stack` is automatically installed by the installation script.
SERVER_DEPENDENCIES = [
    "fastapi",
    "fire",
    "httpx",
    "uvicorn",
]


def stack_apis() -> List[Api]:
    return [v for v in Api]


class AutoRoutedApiInfo(BaseModel):
    routing_table_api: Api
    router_api: Api


def builtin_automatically_routed_apis() -> List[AutoRoutedApiInfo]:
    return [
        AutoRoutedApiInfo(
            routing_table_api=Api.models,
            router_api=Api.inference,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.shields,
            router_api=Api.safety,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.memory_banks,
            router_api=Api.memory,
        ),
    ]


def api_endpoints() -> Dict[Api, List[ApiEndpoint]]:
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


def api_providers() -> Dict[Api, Dict[str, ProviderSpec]]:
    ret = {}
    routing_table_apis = set(
        x.routing_table_api for x in builtin_automatically_routed_apis()
    )
    for api in stack_apis():
        if api in routing_table_apis:
            continue

        name = api.name.lower()
        module = importlib.import_module(f"llama_stack.providers.registry.{name}")
        ret[api] = {
            "remote": remote_provider_spec(api),
            **{a.provider_id: a for a in module.available_providers()},
        }

    return ret
