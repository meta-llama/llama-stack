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
    return [api for api in Api if api not in routing_table_apis and api != Api.inspect]


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
