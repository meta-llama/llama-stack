# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_toolchain.core.datatypes import *  # noqa: F403
from termcolor import cprint

from llama_toolchain.common.prompt_for_config import prompt_for_config
from llama_toolchain.core.distribution import api_providers
from llama_toolchain.core.dynamic import instantiate_class_type


def configure_api_providers(existing_configs: Dict[str, Any]) -> None:
    all_providers = api_providers()

    provider_configs = {}
    for api_str, stub_config in existing_configs.items():
        api = Api(api_str)
        providers = all_providers[api]
        provider_type = stub_config["provider_type"]
        if provider_type not in providers:
            raise ValueError(
                f"Unknown provider `{provider_type}` is not available for API `{api_str}`"
            )

        provider_spec = providers[provider_type]
        cprint(f"Configuring API: {api_str} ({provider_type})", "white", attrs=["bold"])
        config_type = instantiate_class_type(provider_spec.config_class)

        try:
            existing_provider_config = config_type(**stub_config)
        except Exception:
            existing_provider_config = None

        provider_config = prompt_for_config(
            config_type,
            existing_provider_config,
        )
        print("")

        provider_configs[api_str] = {
            "provider_type": provider_type,
            **provider_config.dict(),
        }

    return provider_configs
