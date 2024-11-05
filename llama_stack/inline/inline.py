# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import yaml

from llama_stack.providers.datatypes import *  # noqa: F403
from llama_stack.distribution.datatypes import *  # noqa: F403
from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
from llama_stack.distribution.distribution import get_provider_registry
from llama_stack.distribution.resolver import resolve_impls


class LlamaStackInline:
    def __init__(self, run_config_path: str):
        self.run_config_path = run_config_path
        self.impls = {}
        self.run_config = None

    def print_pip_command(self):
        # TODO: de-dupe this with build.py
        all_providers = get_provider_registry()
        deps = []
        for (
            api_str,
            provider_or_providers,
        ) in self.run_config.providers.items():
            providers_for_api = all_providers[Api(api_str)]

            providers = (
                provider_or_providers
                if isinstance(provider_or_providers, list)
                else [provider_or_providers]
            )

            for provider in providers:
                if provider.provider_id not in providers_for_api:
                    raise ValueError(
                        f"Provider `{provider}` is not available for API `{api_str}`"
                    )

                provider_spec = providers_for_api[provider.provider_id]
                deps.extend(provider_spec.pip_packages)
                if provider_spec.docker_image:
                    raise ValueError(
                        "A stack's dependencies cannot have a docker image"
                    )

        normal_deps = []
        special_deps = []
        for package in deps:
            if "--no-deps" in package or "--index-url" in package:
                special_deps.append(package)
            else:
                normal_deps.append(package)
        deps = list(set(deps))
        special_deps = list(set(special_deps))

        print(
            f"Please install needed dependencies using the following commands:\n\n\tpip install {' '.join(normal_deps)}"
        )
        for special_dep in special_deps:
            print(f"\tpip install {special_dep}")
        print()

    async def initialize(self):
        with open(self.run_config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.run_config = parse_and_maybe_upgrade_config(config_dict)

        all_providers = get_provider_registry()

        try:
            impls = await resolve_impls(self.run_config, all_providers)
            self.impls = impls
        except ModuleNotFoundError as e:
            print(str(e))
            self.print_pip_command()

        if "provider_data" in config_dict:
            provider_id = chosen[api.value][0].provider_id
            provider_data = config_dict["provider_data"].get(provider_id, {})
            if provider_data:
                set_request_provider_data(
                    {"X-LlamaStack-ProviderData": json.dumps(provider_data)}
                )
