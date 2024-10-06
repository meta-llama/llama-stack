# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.datatypes import *  # noqa: F403


class StackRun(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "run",
            prog="llama stack run",
            description="""start the server for a Llama Stack Distribution. You should have already built (or downloaded) and configured the distribution.""",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_run_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "config",
            type=str,
            help="Path to config file to use for the run",
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

    def _run_stack_run_cmd(self, args: argparse.Namespace) -> None:
        from pathlib import Path

        import pkg_resources
        import yaml

        from llama_stack.distribution.build import ImageType
        from llama_stack.distribution.utils.config_dirs import BUILDS_BASE_DIR

        from llama_stack.distribution.utils.exec import run_with_pty

        if not args.config:
            self.parser.error("Must specify a config file to run")
            return

        config_file = Path(args.config)
        if not config_file.exists() and not args.config.endswith(".yaml"):
            # check if it's a build config saved to conda dir
            config_file = Path(
                BUILDS_BASE_DIR / ImageType.conda.value / f"{args.config}-run.yaml"
            )

        if not config_file.exists() and not args.config.endswith(".yaml"):
            # check if it's a build config saved to docker dir
            config_file = Path(
                BUILDS_BASE_DIR / ImageType.docker.value / f"{args.config}-run.yaml"
            )

        if not config_file.exists():
            self.parser.error(
                f"File {str(config_file)} does not exist. Please run `llama stack build` and `llama stack configure <name>` to generate a run.yaml file"
            )
            return

        with open(config_file, "r") as f:
            config = StackRunConfig(**yaml.safe_load(f))

        if config.docker_image:
            script = pkg_resources.resource_filename(
                "llama_stack",
                "distribution/start_container.sh",
            )
            run_args = [script, config.docker_image]
        else:
            script = pkg_resources.resource_filename(
                "llama_stack",
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
