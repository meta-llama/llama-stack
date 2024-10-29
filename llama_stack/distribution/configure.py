# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import textwrap

from typing import Any

from llama_stack.distribution.datatypes import *  # noqa: F403
from termcolor import cprint

from llama_stack.distribution.distribution import (
    builtin_automatically_routed_apis,
    get_provider_registry,
)
from llama_stack.distribution.utils.dynamic import instantiate_class_type
from llama_stack.distribution.utils.prompt_for_config import prompt_for_config


from llama_stack.apis.models import *  # noqa: F403
from llama_stack.apis.shields import *  # noqa: F403
from llama_stack.apis.memory_banks import *  # noqa: F403


def configure_single_provider(
    registry: Dict[str, ProviderSpec], provider: Provider
) -> Provider:
    provider_spec = registry[provider.provider_type]
    config_type = instantiate_class_type(provider_spec.config_class)
    try:
        if provider.config:
            existing = config_type(**provider.config)
        else:
            existing = None
    except Exception:
        existing = None

    cfg = prompt_for_config(config_type, existing)
    return Provider(
        provider_id=provider.provider_id,
        provider_type=provider.provider_type,
        config=cfg.dict(),
    )


def configure_api_providers(
    config: StackRunConfig, build_spec: DistributionSpec
) -> StackRunConfig:
    is_nux = len(config.providers) == 0

    if is_nux:
        print(
            textwrap.dedent(
                """
        Llama Stack is composed of several APIs working together. For each API served by the Stack,
        we need to configure the providers (implementations) you want to use for these APIs.
"""
            )
        )

    provider_registry = get_provider_registry()
    builtin_apis = [a.routing_table_api for a in builtin_automatically_routed_apis()]

    if config.apis:
        apis_to_serve = config.apis
    else:
        apis_to_serve = [a.value for a in Api if a not in (Api.telemetry, Api.inspect)]

    for api_str in apis_to_serve:
        api = Api(api_str)
        if api in builtin_apis:
            continue
        if api not in provider_registry:
            raise ValueError(f"Unknown API `{api_str}`")

        existing_providers = config.providers.get(api_str, [])
        if existing_providers:
            cprint(
                f"Re-configuring existing providers for API `{api_str}`...",
                "green",
                attrs=["bold"],
            )
            updated_providers = []
            for p in existing_providers:
                print(f"> Configuring provider `({p.provider_type})`")
                updated_providers.append(
                    configure_single_provider(provider_registry[api], p)
                )
                print("")
        else:
            # we are newly configuring this API
            plist = build_spec.providers.get(api_str, [])
            plist = plist if isinstance(plist, list) else [plist]

            if not plist:
                raise ValueError(f"No provider configured for API {api_str}?")

            cprint(f"Configuring API `{api_str}`...", "green", attrs=["bold"])
            updated_providers = []
            for i, provider_type in enumerate(plist):
                if i >= 1:
                    others = ", ".join(plist[i:])
                    print(
                        f"Not configuring other providers ({others}) interactively. Please edit the resulting YAML directly.\n"
                    )
                    break

                print(f"> Configuring provider `({provider_type})`")
                updated_providers.append(
                    configure_single_provider(
                        provider_registry[api],
                        Provider(
                            provider_id=(
                                f"{provider_type}-{i:02d}"
                                if len(plist) > 1
                                else provider_type
                            ),
                            provider_type=provider_type,
                            config={},
                        ),
                    )
                )
                print("")

        config.providers[api_str] = updated_providers

    return config


def upgrade_from_routing_table(
    config_dict: Dict[str, Any],
) -> Dict[str, Any]:
    def get_providers(entries):
        return [
            Provider(
                provider_id=(
                    f"{entry['provider_type']}-{i:02d}"
                    if len(entries) > 1
                    else entry["provider_type"]
                ),
                provider_type=entry["provider_type"],
                config=entry["config"],
            )
            for i, entry in enumerate(entries)
        ]

    providers_by_api = {}

    routing_table = config_dict.get("routing_table", {})
    for api_str, entries in routing_table.items():
        providers = get_providers(entries)
        providers_by_api[api_str] = providers

    provider_map = config_dict.get("api_providers", config_dict.get("provider_map", {}))
    if provider_map:
        for api_str, provider in provider_map.items():
            if isinstance(provider, dict) and "provider_type" in provider:
                providers_by_api[api_str] = [
                    Provider(
                        provider_id=f"{provider['provider_type']}",
                        provider_type=provider["provider_type"],
                        config=provider["config"],
                    )
                ]

    config_dict["providers"] = providers_by_api

    config_dict.pop("routing_table", None)
    config_dict.pop("api_providers", None)
    config_dict.pop("provider_map", None)

    config_dict["apis"] = config_dict["apis_to_serve"]
    config_dict.pop("apis_to_serve", None)

    return config_dict


def parse_and_maybe_upgrade_config(config_dict: Dict[str, Any]) -> StackRunConfig:
    version = config_dict.get("version", None)
    if version == LLAMA_STACK_RUN_CONFIG_VERSION:
        return StackRunConfig(**config_dict)

    if "routing_table" in config_dict:
        print("Upgrading config...")
        config_dict = upgrade_from_routing_table(config_dict)

    config_dict["version"] = LLAMA_STACK_RUN_CONFIG_VERSION
    config_dict["built_at"] = datetime.now().isoformat()

    return StackRunConfig(**config_dict)
