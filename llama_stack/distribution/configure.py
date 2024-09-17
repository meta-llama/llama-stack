# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel

from llama_stack.distribution.datatypes import *  # noqa: F403
from termcolor import cprint

from llama_stack.distribution.distribution import api_providers, stack_apis
from llama_stack.distribution.utils.dynamic import instantiate_class_type

from llama_stack.distribution.utils.prompt_for_config import prompt_for_config


# These are hacks so we can re-use the `prompt_for_config` utility
# This needs a bunch of work to be made very user friendly.
class ReqApis(BaseModel):
    apis_to_serve: List[str]


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
    cprint("Configuring APIs to serve...", "white", attrs=["bold"])
    print("Enter comma-separated list of APIs to serve:")

    apis = config.apis_to_serve or list(spec.providers.keys())
    apis = [a for a in apis if a != "telemetry"]
    req_apis = ReqApis(
        apis_to_serve=apis,
    )
    req_apis = prompt_for_config(ReqApis, req_apis)
    config.apis_to_serve = req_apis.apis_to_serve
    print("")

    apis = [v.value for v in stack_apis()]
    all_providers = api_providers()

    apis_to_serve = req_apis.apis_to_serve + ["telemetry"]
    for api_str in apis_to_serve:
        if api_str not in apis:
            raise ValueError(f"Unknown API `{api_str}`")

        cprint(f"Configuring API `{api_str}`...\n", "white", attrs=["bold"])
        api = Api(api_str)
        if isinstance(spec.providers[api_str], list):
            print(
                "You have specified multiple providers for this API. We will configure a routing table now. For each provider, provide a routing key followed by provider configuration.\n"
            )
            routing_entries = []
            for p in spec.providers[api_str]:
                print(f"Configuring provider `{p}`...")
                provider_spec = all_providers[api][p]
                config_type = instantiate_class_type(provider_spec.config_class)

                wrapper_type = make_routing_entry_type(config_type)
                rt_entry = prompt_for_config(wrapper_type, None)

                # TODO: we need to validate the routing keys
                routing_entries.append(
                    ProviderRoutingEntry(
                        provider_id=p,
                        routing_key=rt_entry.routing_key,
                        config=rt_entry.config.dict(),
                    )
                )
            config.provider_map[api_str] = routing_entries
        else:
            p = spec.providers[api_str]
            print(f"Configuring provider `{p}`...")
            provider_spec = all_providers[api][p]
            config_type = instantiate_class_type(provider_spec.config_class)
            cfg = prompt_for_config(config_type, None)
            config.provider_map[api_str] = GenericProviderConfig(
                provider_id=p,
                config=cfg.dict(),
            )

    return config
