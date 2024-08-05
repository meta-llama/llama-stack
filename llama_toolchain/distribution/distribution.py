# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import inspect
from typing import Dict, List

from llama_toolchain.agentic_system.api.endpoints import AgenticSystem
from llama_toolchain.inference.api.endpoints import Inference
from llama_toolchain.safety.api.endpoints import Safety

from .datatypes import Api, ApiEndpoint, Distribution, SourceAdapter


def distribution_dependencies(distribution: Distribution) -> List[str]:
    # only consider SourceAdapters when calculating dependencies
    return [
        dep
        for adapter in distribution.adapters.values()
        if isinstance(adapter, SourceAdapter)
        for dep in adapter.pip_packages
    ] + distribution.additional_pip_packages


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
