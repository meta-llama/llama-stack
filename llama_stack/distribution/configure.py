# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel

from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.distribution import api_providers, stack_apis
from llama_stack.distribution.utils.dynamic import instantiate_class_type

from llama_stack.distribution.utils.prompt_for_config import prompt_for_config
from termcolor import cprint


def make_routing_entry_type(config_class: Any):
    class BaseModelWithConfig(BaseModel):
        routing_key: str
        config: config_class

    return BaseModelWithConfig


def configure_models_api(
    config: StackRunConfig, spec: DistributionSpec
) -> StackRunConfig:
    from llama_stack.providers.impls.builtin.models.config import (
        ModelConfigProviderEntry,
    )
    from prompt_toolkit import prompt

    cprint(f"Configuring API `models`...\n", "white", attrs=["bold"])
    # models do not need prompting, we can use the pre-existing configs to populate the models_config
    provider = spec.providers["models"]
    models_config_list = []

    # TODO (xiyan): we need to clean up configure with models & routers
    # check inference api
    if "inference" in config.apis_to_serve and "inference" in config.provider_map:
        inference_provider_id = config.provider_map["inference"].provider_id
        inference_provider_config = config.provider_map["inference"].config

        if inference_provider_id == "meta-reference":
            core_model_id = inference_provider_config["model"]
        else:
            core_model_id = prompt(
                "Enter model_id your inference is serving",
                default="Meta-Llama3.1-8B-Instruct",
            )
        models_config_list.append(
            ModelConfigProviderEntry(
                api="inference",
                core_model_id=core_model_id,
                provider_id=inference_provider_id,
                config=inference_provider_config,
            )
        )

    # check safety api for models
    if "safety" in config.apis_to_serve and "safety" in config.provider_map:
        safety_provider_id = config.provider_map["safety"].provider_id
        safety_provider_config = config.provider_map["safety"].config

        if safety_provider_id == "meta-reference":
            for model_type in ["llama_guard_shield", "prompt_guard_shield"]:
                if model_type not in safety_provider_config:
                    continue

                core_model_id = safety_provider_config[model_type]["model"]
                models_config_list.append(
                    ModelConfigProviderEntry(
                        api="safety",
                        core_model_id=core_model_id,
                        provider_id=safety_provider_id,
                        config=safety_provider_config,
                    )
                )

    config.provider_map["models"] = GenericProviderConfig(
        provider_id=spec.providers["models"],
        config={"models_config": models_config_list},
    )

    return config


# TODO: make sure we can deal with existing configuration values correctly
# instead of just overwriting them
def configure_api_providers(
    config: StackRunConfig, spec: DistributionSpec
) -> StackRunConfig:
    apis = config.apis_to_serve or list(spec.providers.keys())
    config.apis_to_serve = [a for a in apis if a != "telemetry"]

    apis = [v.value for v in stack_apis()]
    all_providers = api_providers()

    for api_str in spec.providers.keys():
        if api_str not in apis:
            raise ValueError(f"Unknown API `{api_str}`")

        # configure models builtin api last based on existing configs
        if api_str == "models":
            continue

        cprint(f"Configuring API `{api_str}`...\n", "white", attrs=["bold"])
        api = Api(api_str)

        provider_or_providers = spec.providers[api_str]
        if isinstance(provider_or_providers, list) and len(provider_or_providers) > 1:
            print(
                "You have specified multiple providers for this API. We will configure a routing table now. For each provider, provide a routing key followed by provider configuration.\n"
            )

            routing_entries = []
            for p in provider_or_providers:
                print(f"Configuring provider `{p}`...")
                provider_spec = all_providers[api][p]
                config_type = instantiate_class_type(provider_spec.config_class)

                # TODO: we need to validate the routing keys, and
                # perhaps it is better if we break this out into asking
                # for a routing key separately from the associated config
                wrapper_type = make_routing_entry_type(config_type)
                rt_entry = prompt_for_config(wrapper_type, None)

                routing_entries.append(
                    ProviderRoutingEntry(
                        provider_id=p,
                        routing_key=rt_entry.routing_key,
                        config=rt_entry.config.dict(),
                    )
                )
            config.provider_map[api_str] = routing_entries
        else:
            p = (
                provider_or_providers[0]
                if isinstance(provider_or_providers, list)
                else provider_or_providers
            )
            print(f"Configuring provider `{p}`...")
            provider_spec = all_providers[api][p]
            config_type = instantiate_class_type(provider_spec.config_class)
            try:
                provider_config = config.provider_map.get(api_str)
                if provider_config:
                    existing = config_type(**provider_config.config)
                else:
                    existing = None
            except Exception:
                existing = None
            cfg = prompt_for_config(config_type, existing)
            config.provider_map[api_str] = GenericProviderConfig(
                provider_id=p,
                config=cfg.dict(),
            )

    if "models" in config.apis_to_serve:
        config = configure_models_api(config, spec)

    return config
