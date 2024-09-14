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

    def _configure_llama_distribution(self, build_config: BuildConfig):
        from llama_toolchain.common.serialize import EnumEncoder
        from llama_toolchain.core.configure import configure_api_providers

        builds_dir = BUILDS_BASE_DIR / build_config.image_type
        os.makedirs(builds_dir, exist_ok=True)
        package_name = build_config.name.replace("::", "-")
        package_file = builds_dir / f"{package_name}-run.yaml"

        api2providers = build_config.distribution_spec.providers

        stub_config = {
            api_str: {"provider_type": provider}
            for api_str, provider in api2providers.items()
        }

        if package_file.exists():
            cprint(
                f"Configuration already exists for {build_config.distribution}. Will overwrite...",
                "yellow",
                attrs=["bold"],
            )
            config = PackageConfig(**yaml.safe_load(package_file.read_text()))
        else:
            config = PackageConfig(
                built_at=datetime.now(),
                package_name=package_name,
                providers=stub_config,
            )

        config.providers = configure_api_providers(config.providers)
        config.distribution_type = build_config.distribution_spec.distribution_type
        config.docker_image = (
            package_name if build_config.image_type == "docker" else None
        )
        config.conda_env = package_name if build_config.image_type == "conda" else None

        with open(package_file, "w") as f:
            to_write = json.loads(json.dumps(config.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        print(f"YAML configuration has been written to {package_file}")
