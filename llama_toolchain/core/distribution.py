# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
import inspect
from typing import Dict, List

from llama_toolchain.agentic_system.api import AgenticSystem
from llama_toolchain.inference.api import Inference
from llama_toolchain.memory.api import Memory
from llama_toolchain.safety.api import Safety
from llama_toolchain.telemetry.api import Telemetry

from .datatypes import Api, ApiEndpoint, ProviderSpec, remote_provider_spec

# These are the dependencies needed by the distribution server.
# `llama-toolchain` is automatically installed by the installation script.
SERVER_DEPENDENCIES = [
    "fastapi",
    "uvicorn",
]


def stack_apis() -> List[Api]:
    return [v for v in Api]


def api_endpoints() -> Dict[Api, List[ApiEndpoint]]:
    apis = {}

    protocols = {
        Api.inference: Inference,
        Api.safety: Safety,
        Api.agentic_system: AgenticSystem,
        Api.memory: Memory,
        Api.telemetry: Telemetry,
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
    for api in stack_apis():
        name = api.name.lower()
        module = importlib.import_module(f"llama_toolchain.{name}.providers")
        ret[api] = {
            "remote": remote_provider_spec(api),
            **{a.provider_id: a for a in module.available_providers()},
        }

    return ret
