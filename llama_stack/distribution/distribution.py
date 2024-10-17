# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import importlib
from typing import Dict, List

from pydantic import BaseModel

from llama_stack.providers.datatypes import Api, ProviderSpec, remote_provider_spec


def stack_apis() -> List[Api]:
    return [v for v in Api]


class AutoRoutedApiInfo(BaseModel):
    routing_table_api: Api
    router_api: Api


class RegistryApiInfo(BaseModel):
    registry_api: Api
    # registry: Registry


def builtin_registry_apis() -> List[RegistryApiInfo]:
    return [
        RegistryApiInfo(
            registry_api=Api.datasets,
        )
    ]


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


def providable_apis() -> List[Api]:
    routing_table_apis = set(
        x.routing_table_api for x in builtin_automatically_routed_apis()
    )
    registry_apis = set(
        x.registry_api for x in builtin_registry_apis() if x.registry_api
    )
    non_providable_apis = routing_table_apis | registry_apis | {Api.inspect}

    return [api for api in Api if api not in non_providable_apis]


def get_provider_registry() -> Dict[Api, Dict[str, ProviderSpec]]:
    ret = {}
    for api in providable_apis():
        name = api.name.lower()
        module = importlib.import_module(f"llama_stack.providers.registry.{name}")
        ret[api] = {
            "remote": remote_provider_spec(api),
            **{a.provider_type: a for a in module.available_providers()},
        }

    return ret
