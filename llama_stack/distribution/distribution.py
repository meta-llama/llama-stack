# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
from typing import Dict, List

from pydantic import BaseModel

from llama_stack.providers.datatypes import Api, ProviderSpec


def stack_apis() -> List[Api]:
    return list(Api)


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
            routing_table_api=Api.vector_dbs,
            router_api=Api.vector_io,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.datasets,
            router_api=Api.datasetio,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.scoring_functions,
            router_api=Api.scoring,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.benchmarks,
            router_api=Api.eval,
        ),
        AutoRoutedApiInfo(
            routing_table_api=Api.tool_groups,
            router_api=Api.tool_runtime,
        ),
    ]


def providable_apis() -> List[Api]:
    routing_table_apis = {x.routing_table_api for x in builtin_automatically_routed_apis()}
    return [api for api in Api if api not in routing_table_apis and api != Api.inspect and api != Api.providers]


def get_provider_registry() -> Dict[Api, Dict[str, ProviderSpec]]:
    ret = {}
    for api in providable_apis():
        name = api.name.lower()
        module = importlib.import_module(f"llama_stack.providers.registry.{name}")
        ret[api] = {a.provider_type: a for a in module.available_providers()}

    return ret
