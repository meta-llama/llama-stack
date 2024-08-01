# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import shlex
import subprocess

from pathlib import Path

import pkg_resources

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.distribution.registry import all_registered_distributions
from llama_toolchain.utils import LLAMA_STACK_CONFIG_DIR


DISTRIBS_BASE_DIR = Path(LLAMA_STACK_CONFIG_DIR) / "distributions"

DISTRIBS = all_registered_distributions()


class DistributionInstall(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "install",
            prog="llama distribution install",
            description="Install a llama stack distribution",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_distribution_install_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the distribution to install -- (try local-ollama)",
            required=True,
            choices=[d.name for d in DISTRIBS],
        )
        self.parser.add_argument(
            "--conda-env",
            type=str,
            help="Specify the name of the conda environment you would like to create or update",
            required=True,
        )

    def _run_distribution_install_cmd(self, args: argparse.Namespace) -> None:
        os.makedirs(DISTRIBS_BASE_DIR, exist_ok=True)
        script = pkg_resources.resource_filename(
            "llama_toolchain",
            "distribution/install_distribution.sh",
        )

        dist = None
        for d in DISTRIBS:
            if d.name == args.name:
                dist = d
                break

        if dist is None:
            self.parser.error(f"Could not find distribution {args.name}")
            return

        os.makedirs(DISTRIBS_BASE_DIR / dist.name, exist_ok=True)
        run_shell_script(script, args.conda_env, " ".join(dist.pip_packages))
        with open(DISTRIBS_BASE_DIR / dist.name / "conda.env", "w") as f:
            f.write(f"{args.conda_env}\n")


def run_shell_script(script_path, *args):
    command_string = f"{script_path} {' '.join(shlex.quote(str(arg)) for arg in args)}"
    command_list = shlex.split(command_string)
    print(f"Running command: {command_list}")
    subprocess.run(command_list, check=True, text=True)
