# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
from typing import Dict, List

from llama_toolchain.agentic_system.api.endpoints import AgenticSystem
from llama_toolchain.agentic_system.providers import available_agentic_system_providers
from llama_toolchain.inference.api.endpoints import Inference
from llama_toolchain.inference.providers import available_inference_providers
from llama_toolchain.safety.api.endpoints import Safety
from llama_toolchain.safety.providers import available_safety_providers

from .datatypes import (
    Api,
    ApiEndpoint,
    DistributionSpec,
    InlineProviderSpec,
    ProviderSpec,
)

# These are the dependencies needed by the distribution server.
# `llama-toolchain` is automatically installed by the installation script.
SERVER_DEPENDENCIES = [
    "fastapi",
    "python-dotenv",
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


def api_endpoints() -> Dict[Api, List[ApiEndpoint]]:
    apis = {}

    protocols = {
        Api.inference: Inference,
        Api.safety: Safety,
        Api.agentic_system: AgenticSystem,
    }

    for api, protocol in protocols.items():
        endpoints = []
        protocol_methods = inspect.getmembers(protocol, predicate=inspect.isfunction)

        for name, method in protocol_methods:
            if not hasattr(method, "__webmethod__"):
                continue

            webmethod = method.__webmethod__
            route = webmethod.route

            # use `post` for all methods right now until we fix up the `webmethod` openapi
            # annotation and write our own openapi generator
            endpoints.append(ApiEndpoint(route=route, method="post", name=name))

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

    return {
        Api.inference: inference_providers_by_id,
        Api.safety: safety_providers_by_id,
        Api.agentic_system: agentic_system_providers_by_id,
    }
