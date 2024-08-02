# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import shlex

import pkg_resources

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.distribution.distribution import distribution_dependencies
from llama_toolchain.distribution.registry import (
    available_distributions,
    resolve_distribution,
)
from llama_toolchain.utils import DISTRIBS_BASE_DIR

from .utils import run_command, run_with_pty

DISTRIBS = available_distributions()


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

        dist = resolve_distribution(args.name)
        if dist is None:
            self.parser.error(f"Could not find distribution {args.name}")
            return

        os.makedirs(DISTRIBS_BASE_DIR / dist.name, exist_ok=True)

        deps = distribution_dependencies(dist)
        run_command([script, args.conda_env, " ".join(deps)])
        with open(DISTRIBS_BASE_DIR / dist.name / "conda.env", "w") as f:
            f.write(f"{args.conda_env}\n")

        # we need to run configure _within_ the conda environment and need to run with
        # a pty since configure is
        python_exe = run_command(
            shlex.split(f"conda run -n {args.conda_env} which python")
        ).strip()
        run_with_pty(
            shlex.split(
                f"{python_exe} -m llama_toolchain.cli.llama distribution configure --name {dist.name}"
            )
        )
