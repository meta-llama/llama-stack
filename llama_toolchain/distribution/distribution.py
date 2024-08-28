# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
from typing import Dict, List

from llama_toolchain.agentic_system.api import AgenticSystem
from llama_toolchain.agentic_system.providers import available_agentic_system_providers
from llama_toolchain.inference.api import Inference
from llama_toolchain.inference.providers import available_inference_providers
from llama_toolchain.memory.api import Memory
from llama_toolchain.memory.providers import available_memory_providers
from llama_toolchain.safety.api import Safety
from llama_toolchain.safety.providers import available_safety_providers

from .datatypes import (
    Api,
    ApiEndpoint,
    DistributionSpec,
    InlineProviderSpec,
    ProviderSpec,
    remote_provider_spec,
)

# These are the dependencies needed by the distribution server.
# `llama-toolchain` is automatically installed by the installation script.
SERVER_DEPENDENCIES = [
    "fastapi",
    "uvicorn",
]


def distribution_dependencies(distribution: DistributionSpec) -> List[str]:
    # only consider InlineProviderSpecs when calculating dependencies
    return [
        dep
        for provider_spec in distribution.provider_specs.values()
        if isinstance(provider_spec, InlineProviderSpec)
        for dep in provider_spec.pip_packages
    ] + SERVER_DEPENDENCIES


def stack_apis() -> List[Api]:
    return [Api.inference, Api.safety, Api.agentic_system, Api.memory]


def api_endpoints() -> Dict[Api, List[ApiEndpoint]]:
    apis = {}

    protocols = {
        Api.inference: Inference,
        Api.safety: Safety,
        Api.agentic_system: AgenticSystem,
        Api.memory: Memory,
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
    inference_providers_by_id = {
        a.provider_id: a for a in available_inference_providers()
    }
    safety_providers_by_id = {a.provider_id: a for a in available_safety_providers()}
    agentic_system_providers_by_id = {
        a.provider_id: a for a in available_agentic_system_providers()
    }

    ret = {
        Api.inference: inference_providers_by_id,
        Api.safety: safety_providers_by_id,
        Api.agentic_system: agentic_system_providers_by_id,
        Api.memory: {a.provider_id: a for a in available_memory_providers()},
    }
    for k, v in ret.items():
        v["remote"] = remote_provider_spec(k)
    return ret
