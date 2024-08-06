# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import shlex
import textwrap

import pkg_resources

from termcolor import cprint

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import DISTRIBS_BASE_DIR


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
        from llama_toolchain.distribution.registry import available_distributions

        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the distribution to install -- (try local-ollama)",
            required=True,
            choices=[d.name for d in available_distributions()],
        )
        self.parser.add_argument(
            "--conda-env",
            type=str,
            help="Specify the name of the conda environment you would like to create or update",
            required=True,
        )

    def _run_distribution_install_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.common.exec import run_with_pty
        from llama_toolchain.distribution.distribution import distribution_dependencies
        from llama_toolchain.distribution.registry import resolve_distribution

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
        return_code = run_with_pty([script, args.conda_env, " ".join(deps)])

        assert return_code == 0, cprint(
            f"Failed to install distribution {dist.name}", color="red"
        )

        with open(DISTRIBS_BASE_DIR / dist.name / "conda.env", "w") as f:
            f.write(f"{args.conda_env}\n")

        cprint(
            f"Distribution `{dist.name}` has been installed successfully!",
            color="green",
        )
        print(
            textwrap.dedent(
                f"""
                Update your conda environment and configure this distribution by running:

                conda deactivate && conda activate {args.conda_env}
                llama distribution configure --name {dist.name}
            """
            )
        )
