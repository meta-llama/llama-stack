# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
from pathlib import Path

import pkg_resources

import yaml
from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import BUILDS_BASE_DIR

from llama_toolchain.common.exec import run_with_pty
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
        self.parser.add_argument(
            "config",
            type=str,
            help="Path to the build config file (e.g. ~/.llama/builds/<image_type>/<name>-build.yaml). For docker, this could also be the name of the docker image. ",
        )

    def _run_stack_configure_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.core.package import ImageType

        build_config_file = Path(args.config)
        if not build_config_file.exists():
            cprint(
                f"Could not find {build_config_file}. Trying docker image name instead...",
                color="green",
            )

            build_dir = (
                Path(os.path.expanduser("./.llama/distributions"))
                / ImageType.docker.value
            )
            build_config_file = build_dir / f"{args.config}-build.yaml"

            os.makedirs(build_dir, exist_ok=True)

            script = pkg_resources.resource_filename(
                "llama_toolchain", "core/configure_container.sh"
            )
            script_args = [
                script,
                args.config,
                str(build_config_file),
            ]

            return_code = run_with_pty(script_args)

            # we have regenerated the build config file with script, now check if it exists
            build_config_file = Path(str(build_config_file))
            if return_code != 0 or not build_config_file.exists():
                self.parser.error(
                    f"Can not find {build_config_file}. Please run llama stack build first or check if docker image exists"
                )
                return

        with open(build_config_file, "r") as f:
            build_config = BuildConfig(**yaml.safe_load(f))

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
                f"Configuration already exists for {build_config.name}. Will overwrite...",
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
