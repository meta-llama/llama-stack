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
from termcolor import cprint


def make_routing_entry_type(config_class: Any):
    class BaseModelWithConfig(BaseModel):
        routing_key: str
        config: config_class

    return BaseModelWithConfig


# TODO: make sure we can deal with existing configuration values correctly
# instead of just overwriting them
def configure_api_providers(
    config: StackRunConfig, spec: DistributionSpec
) -> StackRunConfig:
    cprint(f"configure_api_providers {spec}", "red")
    apis = config.apis_to_serve or list(spec.providers.keys())

    # append the bulitin automatically routed APIs
    for inf in builtin_automatically_routed_apis():
        if inf.router_api.value in apis:
            apis.append(inf.routing_table_api.value)

    config.apis_to_serve = [a for a in apis if a != "telemetry"]

    apis = [v.value for v in stack_apis()]
    all_providers = api_providers()

    for api_str in spec.providers.keys():
        if api_str not in apis:
            raise ValueError(f"Unknown API `{api_str}`")

        cprint(f"Configuring API `{api_str}`...\n", "white", attrs=["bold"])
        api = Api(api_str)

        provider_or_providers = spec.providers[api_str]
        p = (
            provider_or_providers[0]
            if isinstance(provider_or_providers, list)
            else provider_or_providers
        )
        print(f"Configuring provider `{p}`...")
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
        config.api_providers[api_str] = GenericProviderConfig(
            provider_id=p,
            config=cfg.dict(),
        )

    return config
