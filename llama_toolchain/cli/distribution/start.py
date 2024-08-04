# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import shlex
from pathlib import Path

import yaml

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import DISTRIBS_BASE_DIR


class DistributionStart(Subcommand):

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "start",
            prog="llama distribution start",
            description="""start the server for a Llama stack distribution. you should have already installed and configured the distribution""",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_distribution_start_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the distribution to start",
            required=True,
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

    def _run_distribution_start_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.common.exec import run_command
        from llama_toolchain.distribution.registry import resolve_distribution
        from llama_toolchain.distribution.server import main as distribution_server_init

        dist = resolve_distribution(args.name)
        if dist is None:
            self.parser.error(f"Distribution with name {args.name} not found")
            return

        config_yaml = Path(DISTRIBS_BASE_DIR) / dist.name / "config.yaml"
        if not config_yaml.exists():
            raise ValueError(
                f"Configuration {config_yaml} does not exist. Please run `llama distribution install` or `llama distribution configure` first"
            )

        with open(config_yaml, "r") as fp:
            config = yaml.safe_load(fp)

        conda_env = config["conda_env"]

        python_exe = run_command(shlex.split("which python"))
        # simple check, unfortunate
        if conda_env not in python_exe:
            raise ValueError(
                f"Please re-run start after activating the `{conda_env}` conda environment first"
            )

        distribution_server_init(
            dist.name,
            config_yaml,
            args.port,
            disable_ipv6=args.disable_ipv6,
        )
