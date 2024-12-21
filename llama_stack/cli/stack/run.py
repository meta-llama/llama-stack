# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from pathlib import Path

from llama_stack.cli.subcommand import Subcommand

REPO_ROOT = Path(__file__).parent.parent.parent.parent


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
        self.parser.add_argument(
            "--env",
            action="append",
            help="Environment variables to pass to the server in KEY=VALUE format. Can be specified multiple times.",
            default=[],
            metavar="KEY=VALUE",
        )

    def _run_stack_run_cmd(self, args: argparse.Namespace) -> None:
        import importlib.resources
        import yaml

        from llama_stack.distribution.build import ImageType
        from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
        from llama_stack.distribution.utils.config_dirs import (
            BUILDS_BASE_DIR,
            DISTRIBS_BASE_DIR,
        )
        from llama_stack.distribution.utils.exec import run_with_pty

        if not args.config:
            self.parser.error("Must specify a config file to run")
            return

        config_file = Path(args.config)
        has_yaml_suffix = args.config.endswith(".yaml")

        if not config_file.exists() and not has_yaml_suffix:
            # check if this is a template
            config_file = (
                Path(REPO_ROOT) / "llama_stack" / "templates" / args.config / "run.yaml"
            )

        if not config_file.exists() and not has_yaml_suffix:
            # check if it's a build config saved to conda dir
            config_file = Path(
                BUILDS_BASE_DIR / ImageType.conda.value / f"{args.config}-run.yaml"
            )

        if not config_file.exists() and not has_yaml_suffix:
            # check if it's a build config saved to docker dir
            config_file = Path(
                BUILDS_BASE_DIR / ImageType.docker.value / f"{args.config}-run.yaml"
            )

        if not config_file.exists() and not has_yaml_suffix:
            # check if it's a build config saved to ~/.llama dir
            config_file = Path(
                DISTRIBS_BASE_DIR
                / f"llamastack-{args.config}"
                / f"{args.config}-run.yaml"
            )

        if not config_file.exists():
            self.parser.error(
                f"File {str(config_file)} does not exist. Please run `llama stack build` to generate (and optionally edit) a run.yaml file"
            )
            return

        print(f"Using config file: {config_file}")
        config_dict = yaml.safe_load(config_file.read_text())
        config = parse_and_maybe_upgrade_config(config_dict)

        if config.docker_image:
            script = (
                importlib.resources.files("llama_stack")
                / "distribution/start_container.sh"
            )
            run_args = [script, config.docker_image]
        else:
            script = (
                importlib.resources.files("llama_stack")
                / "distribution/start_conda_env.sh"
            )
            run_args = [
                script,
                config.conda_env,
            ]

        run_args.extend([str(config_file), str(args.port)])
        if args.disable_ipv6:
            run_args.append("--disable-ipv6")

        for env_var in args.env:
            if "=" not in env_var:
                self.parser.error(
                    f"Environment variable '{env_var}' must be in KEY=VALUE format"
                )
                return
            key, value = env_var.split("=", 1)  # split on first = only
            if not key:
                self.parser.error(f"Environment variable '{env_var}' has empty key")
                return
            run_args.extend(["--env", f"{key}={value}"])

        run_with_pty(run_args)
