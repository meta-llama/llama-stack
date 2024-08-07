# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import textwrap

import pkg_resources
import yaml

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
        from llama_toolchain.distribution.registry import available_distribution_specs

        self.parser.add_argument(
            "--spec",
            type=str,
            help="Distribution spec to install (try ollama-inline)",
            required=True,
            choices=[d.spec_id for d in available_distribution_specs()],
        )
        self.parser.add_argument(
            "--name",
            type=str,
            help="What should the installation be called locally?",
            required=True,
        )
        self.parser.add_argument(
            "--conda-env",
            type=str,
            help="conda env in which this distribution will run (default = distribution name)",
        )

    def _run_distribution_install_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.common.exec import run_with_pty
        from llama_toolchain.distribution.datatypes import DistributionConfig
        from llama_toolchain.distribution.distribution import distribution_dependencies
        from llama_toolchain.distribution.registry import resolve_distribution_spec

        os.makedirs(DISTRIBS_BASE_DIR, exist_ok=True)
        script = pkg_resources.resource_filename(
            "llama_toolchain",
            "distribution/install_distribution.sh",
        )

        dist = resolve_distribution_spec(args.spec)
        if dist is None:
            self.parser.error(f"Could not find distribution {args.spec}")
            return

        distrib_dir = DISTRIBS_BASE_DIR / args.name
        os.makedirs(distrib_dir, exist_ok=True)

        deps = distribution_dependencies(dist)
        if not args.conda_env:
            print(f"Using {args.name} as the Conda environment for this distribution")

        conda_env = args.conda_env or args.name
        return_code = run_with_pty([script, conda_env, " ".join(deps)])

        assert return_code == 0, cprint(
            f"Failed to install distribution {dist.spec_id}", color="red"
        )

        config_file = distrib_dir / "config.yaml"
        with open(config_file, "w") as f:
            c = DistributionConfig(
                spec=dist.spec_id,
                name=args.name,
                conda_env=conda_env,
            )
            f.write(yaml.dump(c.dict(), sort_keys=False))

        cprint(
            f"Distribution `{args.name}` (with spec {dist.spec_id}) has been installed successfully!",
            color="green",
        )
        print(
            textwrap.dedent(
                f"""
                Update your conda environment and configure this distribution by running:

                conda deactivate && conda activate {conda_env}
                llama distribution configure --name {args.name}
            """
            )
        )
