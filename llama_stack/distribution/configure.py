# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.distribution import (
    api_providers,
    builtin_automatically_routed_apis,
    stack_apis,
)
from llama_stack.distribution.utils.dynamic import instantiate_class_type

from llama_stack.distribution.utils.prompt_for_config import prompt_for_config
from prompt_toolkit import prompt
from termcolor import cprint


def make_routing_entry_type(config_class: Any):
    class BaseModelWithConfig(BaseModel):
        routing_key: str
        config: config_class

    return BaseModelWithConfig


def get_builtin_apis(provider_backed_apis: List[str]) -> List[str]:
    """Get corresponding builtin APIs given provider backed APIs"""
    res = []
    for inf in builtin_automatically_routed_apis():
        if inf.router_api.value in provider_backed_apis:
            res.append(inf.routing_table_api.value)

    return res


# TODO: make sure we can deal with existing configuration values correctly
# instead of just overwriting them
def configure_api_providers(
    config: StackRunConfig, spec: DistributionSpec
) -> StackRunConfig:
    cprint(f"configure_api_providers {spec}", "red")
    apis = config.apis_to_serve or list(spec.providers.keys())
    # append the bulitin routing APIs
    apis += get_builtin_apis(apis)

    router_api2builtin_api = {
        inf.router_api.value: inf.routing_table_api.value
        for inf in builtin_automatically_routed_apis()
    }

    config.apis_to_serve = [a for a in apis if a != "telemetry"]

    apis = [v.value for v in stack_apis()]
    all_providers = api_providers()

    # configure simple case for with non-routing providers to api_providers
    for api_str in spec.providers.keys():
        if api_str not in apis:
            raise ValueError(f"Unknown API `{api_str}`")

        cprint(f"Configuring API `{api_str}`...", "green", attrs=["bold"])
        api = Api(api_str)

        p = spec.providers[api_str]
        cprint(f"=== Configuring provider `{p}` for API {api_str}...", "green")

        provider_spec = all_providers[api][p]
        config_type = instantiate_class_type(provider_spec.config_class)
        try:
            provider_config = config.api_providers.get(api_str)
            if provider_config:
                existing = config_type(**provider_config.config)
            else:
                existing = None
        except Exception:
            existing = None
        cfg = prompt_for_config(config_type, existing)

        if api_str in router_api2builtin_api:
            # a routing api, we need to infer and assign it a routing_key and put it in the routing_table
            routing_key = prompt(
                "> Enter routing key for the {} provider: ".format(api_str),
            )
            config.routing_table[]
        else:
            config.api_providers[api_str] = GenericProviderConfig(
                provider_id=p,
                config=cfg.dict(),
            )

        print("")

    return config
