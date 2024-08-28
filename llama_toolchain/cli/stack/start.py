# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from pathlib import Path

import pkg_resources
import yaml

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.distribution.datatypes import *  # noqa: F403


class StackStart(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "start",
            prog="llama stack start",
            description="""start the server for a Llama Stack Distribution. You should have already built (or downloaded) and configured the distribution.""",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_start_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "yaml_config",
            type=str,
            help="Yaml config containing the API build configuration",
        )
        self.parser.add_argument(
            "--port",
            type=int,
            help="Port to run the server on. Defaults to 5000",
            default=5000,
        )
        self.parser.add_argument(
            "--disable-ipv6",
            action="store_true",
            help="Disable IPv6 support",
            default=False,
        )

    def _run_stack_start_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.common.exec import run_with_pty

        config_file = Path(args.yaml_config)
        if not config_file.exists():
            self.parser.error(
                f"Could not find {config_file}. Please run `llama stack build` first"
            )
            return

        with open(config_file, "r") as f:
            config = PackageConfig(**yaml.safe_load(f))

        if not config.distribution_id:
            # this is technically not necessary. everything else continues to work,
            # but maybe we want to be very clear for the users
            self.parser.error(
                "No distribution_id found. Did you want to start a provider?"
            )
            return

        if config.docker_image:
            script = pkg_resources.resource_filename(
                "llama_toolchain",
                "distribution/start_container.sh",
            )
            run_args = [script, config.docker_image]
        else:
            script = pkg_resources.resource_filename(
                "llama_toolchain",
                "distribution/start_conda_env.sh",
            )
            run_args = [
                script,
                config.conda_env,
            ]

        run_args.extend([str(config_file), str(args.port)])
        if args.disable_ipv6:
            run_args.append("--disable-ipv6")

        run_with_pty(run_args)
