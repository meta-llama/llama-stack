# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from llama_models.sku_list import (
    llama3_1_family,
    llama3_2_family,
    llama3_family,
    resolve_model,
    safety_models,
)

from pydantic import BaseModel
from llama_stack.distribution.datatypes import *  # noqa: F403
from prompt_toolkit import prompt
from prompt_toolkit.validation import Validator
from termcolor import cprint

from llama_stack.apis.memory.memory import MemoryBankType
from llama_stack.distribution.distribution import (
    builtin_automatically_routed_apis,
    get_provider_registry,
    stack_apis,
)
from llama_stack.distribution.utils.dynamic import instantiate_class_type

from llama_stack.distribution.utils.prompt_for_config import prompt_for_config
from llama_stack.providers.impls.meta_reference.safety.config import (
    MetaReferenceShieldType,
)


ALLOWED_MODELS = (
    llama3_family() + llama3_1_family() + llama3_2_family() + safety_models()
)


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
    apis = config.apis_to_serve or list(spec.providers.keys())
    # append the bulitin routing APIs
    apis += get_builtin_apis(apis)

    router_api2builtin_api = {
        inf.router_api.value: inf.routing_table_api.value
        for inf in builtin_automatically_routed_apis()
    }

    config.apis_to_serve = list(set([a for a in apis if a != "telemetry"]))

    apis = [v.value for v in stack_apis()]
    all_providers = get_provider_registry()

    # configure simple case for with non-routing providers to api_providers
    for api_str in spec.providers.keys():
        if api_str not in apis:
            raise ValueError(f"Unknown API `{api_str}`")

        cprint(f"Configuring API `{api_str}`...", "green", attrs=["bold"])
        api = Api(api_str)

        p = spec.providers[api_str]
        cprint(f"=== Configuring provider `{p}` for API {api_str}...", "green")

        if isinstance(p, list):
            cprint(
                f"[WARN] Interactive configuration of multiple providers {p} is not supported, configuring {p[0]} only, please manually configure {p[1:]} in routing_table of run.yaml",
                "yellow",
            )
            p = p[0]

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
            routing_key = "<PLEASE_FILL_ROUTING_KEY>"
            routing_entries = []
            if api_str == "inference":
                if hasattr(cfg, "model"):
                    routing_key = cfg.model
                else:
                    routing_key = prompt(
                        "> Please enter the supported model your provider has for inference: ",
                        default="Llama3.1-8B-Instruct",
                        validator=Validator.from_callable(
                            lambda x: resolve_model(x) is not None,
                            error_message="Model must be: {}".format(
                                [x.descriptor() for x in ALLOWED_MODELS]
                            ),
                        ),
                    )
                routing_entries.append(
                    RoutableProviderConfig(
                        routing_key=routing_key,
                        provider_type=p,
                        config=cfg.dict(),
                    )
                )

            if api_str == "safety":
                # TODO: add support for other safety providers, and simplify safety provider config
                if p == "meta-reference":
                    routing_entries.append(
                        RoutableProviderConfig(
                            routing_key=[s.value for s in MetaReferenceShieldType],
                            provider_type=p,
                            config=cfg.dict(),
                        )
                    )
                else:
                    cprint(
                        f"[WARN] Interactive configuration of safety provider {p} is not supported. Please look for `{routing_key}` in run.yaml and replace it appropriately.",
                        "yellow",
                        attrs=["bold"],
                    )
                    routing_entries.append(
                        RoutableProviderConfig(
                            routing_key=routing_key,
                            provider_type=p,
                            config=cfg.dict(),
                        )
                    )

            if api_str == "memory":
                bank_types = list([x.value for x in MemoryBankType])
                routing_key = prompt(
                    "> Please enter the supported memory bank type your provider has for memory: ",
                    default="vector",
                    validator=Validator.from_callable(
                        lambda x: x in bank_types,
                        error_message="Invalid provider, please enter one of the following: {}".format(
                            bank_types
                        ),
                    ),
                )
                routing_entries.append(
                    RoutableProviderConfig(
                        routing_key=routing_key,
                        provider_type=p,
                        config=cfg.dict(),
                    )
                )

            config.routing_table[api_str] = routing_entries
            config.api_providers[api_str] = PlaceholderProviderConfig(
                providers=p if isinstance(p, list) else [p]
            )
        else:
            config.api_providers[api_str] = GenericProviderConfig(
                provider_type=p,
                config=cfg.dict(),
            )

        print("")

    return config
