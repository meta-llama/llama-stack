# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
from pathlib import Path

import yaml

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import BUILDS_BASE_DIR
from termcolor import cprint
from llama_toolchain.core.datatypes import *  # noqa: F403
import os


class StackConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama stack configure",
            description="configure a llama stack distribution",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_configure_cmd)

    def _add_arguments(self):
        from llama_toolchain.core.distribution_registry import (
            available_distribution_specs,
        )
        from llama_toolchain.core.package import ImageType

        allowed_ids = [d.distribution_type for d in available_distribution_specs()]
        self.parser.add_argument(
            "config",
            type=str,
            help="Path to the build config file (e.g. ~/.llama/builds/<image_type>/<name>-build.yaml)",
        )

    def _run_stack_configure_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.core.package import ImageType

        with open(args.config, "r") as f:
            try:
                build_config = BuildConfig(**yaml.safe_load(f))
            except Exception as e:
                self.parser.error(
                    f"Could not find {config_file}. Please run `llama stack build` first"
                )
                return

        self._configure_llama_distribution(build_config)

    def _configure_llama_distribution(self, build_config: BuildConfig) -> None:
        from llama_toolchain.common.serialize import EnumEncoder
        from llama_toolchain.core.configure import configure_api_providers
        from llama_toolchain.core.distribution import api_providers
        from llama_toolchain.core.distribution_registry import resolve_distribution_spec
        from llama_toolchain.core.package import ApiInput

        # build api_inputs from build_config
        # TODO (xiyan): refactor and clean this up with removing distribution type
        api_inputs = []
        if build_config.distribution == "adhoc":
            if not build_config.api_providers:
                self.parser.error(
                    "You must specify API providers with (api=provider,...) for building an adhoc distribution"
                )
                return

            parsed = parse_api_provider_tuples(build_config.api_providers, self.parser)
            for api, provider_spec in parsed.items():
                for dep in provider_spec.api_dependencies:
                    if dep not in parsed:
                        self.parser.error(
                            f"API {api} needs dependency {dep} provided also"
                        )
                        return

                api_inputs.append(
                    ApiInput(
                        api=api,
                        provider=provider_spec.provider_type,
                    )
                )
            docker_image = None
        else:
            if build_config.api_providers:
                self.parser.error(
                    "You cannot specify API providers for pre-registered distributions"
                )
                return

            dist = resolve_distribution_spec(build_config.distribution)
            if dist is None:
                self.parser.error(
                    f"Could not find distribution {build_config.distribution}"
                )
                return

            for api, provider_type in dist.providers.items():
                api_inputs.append(
                    ApiInput(
                        api=api,
                        provider=provider_type,
                    )
                )
            docker_image = dist.docker_image

        # build or get package config
        all_providers = api_providers()

        stub_config = {}
        for api_input in api_inputs:
            api = api_input.api
            providers_for_api = all_providers[api]
            if api_input.provider not in providers_for_api:
                raise ValueError(
                    f"Provider `{api_input.provider}` is not available for API `{api}`"
                )

            provider = providers_for_api[api_input.provider]
            if provider.docker_image:
                raise ValueError("A stack's dependencies cannot have a docker image")

            stub_config[api.value] = {"provider_type": api_input.provider}

        build_dir = (
            BUILDS_BASE_DIR / build_config.distribution / build_config.image_type
        )
        os.makedirs(build_dir, exist_ok=True)
        package_name = build_config.name.replace("::", "-")
        package_file = build_dir / f"{package_name}.yaml"

        if package_file.exists():
            cprint(
                f"Configuration already exists for {build_config.distribution}. Will overwrite...",
                "yellow",
                attrs=["bold"],
            )
            config = PackageConfig(**yaml.safe_load(package_file.read_text()))
            for api_str, new_config in stub_config.items():
                if api_str not in config.providers:
                    config.providers[api_str] = new_config
                else:
                    existing_config = config.providers[api_str]
                    if existing_config["provider_type"] != new_config["provider_type"]:
                        cprint(
                            f"Provider `{api_str}` has changed from `{existing_config}` to `{new_config}`",
                            color="yellow",
                        )
                        config.providers[api_str] = new_config
        else:
            config = PackageConfig(
                built_at=datetime.now(),
                package_name=package_name,
                providers=stub_config,
            )

        config.distribution_type = build_config.distribution
        config.docker_image = (
            package_name if build_config.image_type == "docker" else None
        )
        config.conda_env = package_name if build_config.image_type == "conda" else None

        cprint(
            f"Target `{package_name}` built with configuration at {str(package_file)}",
            color="green",
        )

        config.providers = configure_api_providers(config.providers)

        with open(package_file, "w") as f:
            to_write = json.loads(json.dumps(config.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        print(f"YAML configuration has been written to {package_file}")
