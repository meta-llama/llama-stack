# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os

import pkg_resources
import yaml

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import DISTRIBS_BASE_DIR

from termcolor import cprint


class ApiBuild(Subcommand):

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "install",
            prog="llama api build",
            description="Build a Llama stack API provider container",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_api_build_command)

    def _add_arguments(self):
        from llama_toolchain.distribution.distribution import stack_apis

        allowed_args = [a.name for a in stack_apis()]
        self.parser.add_argument(
            "api",
            choices=allowed_args,
            help="Stack API (one of: {})".format(", ".join(allowed_args))
        )

        self.parser.add_argument(
            "--provider",
            type=str,
            help="The provider to package into the container",
            required=True,
        )
        self.parser.add_argument(
            "--container-name",
            type=str,
            help="Name of the container (including tag if needed)",
            required=True,
        )
        self.parser.add_argument(
            "--dependencies",
            type=str,
            help="Comma separated list of (downstream_api=provider) dependencies needed for the API",
            required=False,
        )

    def _run_api_build_command(self, args: argparse.Namespace) -> None:
        from llama_toolchain.common.exec import run_with_pty
        from llama_toolchain.distribution.datatypes import DistributionConfig
        from llama_toolchain.distribution.distribution import distribution_dependencies
        from llama_toolchain.distribution.registry import resolve_distribution_spec

        os.makedirs(DISTRIBS_BASE_DIR, exist_ok=True)
        script = pkg_resources.resource_filename(
            "llama_toolchain",
            "distribution/build_api.sh",
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

        config_file = distrib_dir / "config.yaml"
        if config_file.exists():
            c = DistributionConfig(**yaml.safe_load(config_file.read_text()))
            if c.spec != dist.spec_id:
                self.parser.error(
                    f"already installed distribution with `spec={c.spec}` does not match provided spec `{args.spec}`"
                )
                return
            if c.conda_env != conda_env:
                self.parser.error(
                    f"already installed distribution has `conda_env={c.conda_env}` different from provided conda env `{conda_env}`"
                )
                return
        else:
            with open(config_file, "w") as f:
                c = DistributionConfig(
                    spec=dist.spec_id,
                    name=args.name,
                    conda_env=conda_env,
                )
                f.write(yaml.dump(c.dict(), sort_keys=False))

        return_code = run_with_pty([script, conda_env, args.name, " ".join(deps)])

        assert return_code == 0, cprint(
            f"Failed to install distribution {dist.spec_id}", color="red"
        )
        cprint(
            f"Distribution `{args.name}` (with spec {dist.spec_id}) has been installed successfully!",
            color="green",
        )
