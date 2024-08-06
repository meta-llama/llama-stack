# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import shlex

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
        from llama_toolchain.distribution.registry import resolve_distribution_spec
        from llama_toolchain.distribution.server import main as distribution_server_init

        config_file = DISTRIBS_BASE_DIR / args.name / "config.yaml"
        if not config_file.exists():
            self.parser.error(
                f"Could not find {config_file}. Please run `llama distribution install` first"
            )
            return

        # we need to find the spec from the name
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        dist = resolve_distribution_spec(config["spec"])
        if dist is None:
            raise ValueError(f"Could not find any registered spec `{config['spec']}`")

        conda_env = config["conda_env"]

        python_exe = run_command(shlex.split("which python"))
        # simple check, unfortunate
        if conda_env not in python_exe:
            raise ValueError(
                f"Please re-run start after activating the `{conda_env}` conda environment first"
            )

        distribution_server_init(
            config_file,
            args.port,
            disable_ipv6=args.disable_ipv6,
        )
