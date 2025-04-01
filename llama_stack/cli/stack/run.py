# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
from pathlib import Path

from llama_stack.cli.stack.utils import ImageType
from llama_stack.cli.subcommand import Subcommand
from llama_stack.log import get_logger

REPO_ROOT = Path(__file__).parent.parent.parent.parent

logger = get_logger(name=__name__, category="server")


class StackRun(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "run",
            prog="llama stack run",
            description="""Start the server for a Llama Stack Distribution. You should have already built (or downloaded) and configured the distribution.""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
            help="Port to run the server on. It can also be passed via the env var LLAMA_STACK_PORT.",
            default=int(os.getenv("LLAMA_STACK_PORT", 8321)),
        )
        self.parser.add_argument(
            "--image-name",
            type=str,
            default=os.environ.get("CONDA_DEFAULT_ENV"),
            help="Name of the image to run. Defaults to the current conda environment",
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
            metavar="KEY=VALUE",
        )
        self.parser.add_argument(
            "--tls-keyfile",
            type=str,
            help="Path to TLS key file for HTTPS",
        )
        self.parser.add_argument(
            "--tls-certfile",
            type=str,
            help="Path to TLS certificate file for HTTPS",
        )
        self.parser.add_argument(
            "--image-type",
            type=str,
            help="Image Type used during the build. This can be either conda or container or venv.",
            choices=[e.value for e in ImageType],
        )

    # If neither image type nor image name is provided, but at the same time
    # the current environment has conda breadcrumbs, then assume what the user
    # wants to use conda mode and not the usual default mode (using
    # pre-installed system packages).
    #
    # Note: yes, this is hacky. It's implemented this way to keep the existing
    # conda users unaffected by the switch of the default behavior to using
    # system packages.
    def _get_image_type_and_name(self, args: argparse.Namespace) -> tuple[str, str]:
        conda_env = os.environ.get("CONDA_DEFAULT_ENV")
        if conda_env and args.image_name == conda_env:
            logger.warning(f"Conda detected. Using conda environment {conda_env} for the run.")
            return ImageType.CONDA.value, args.image_name
        return args.image_type, args.image_name

    def _run_stack_run_cmd(self, args: argparse.Namespace) -> None:
        import yaml

        from llama_stack.distribution.configure import parse_and_maybe_upgrade_config
        from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
        from llama_stack.distribution.utils.exec import formulate_run_args, run_command

        config_file = Path(args.config)
        has_yaml_suffix = args.config.endswith(".yaml")
        template_name = None

        if not config_file.exists() and not has_yaml_suffix:
            # check if this is a template
            config_file = Path(REPO_ROOT) / "llama_stack" / "templates" / args.config / "run.yaml"
            if config_file.exists():
                template_name = args.config

        if not config_file.exists() and not has_yaml_suffix:
            # check if it's a build config saved to ~/.llama dir
            config_file = Path(DISTRIBS_BASE_DIR / f"llamastack-{args.config}" / f"{args.config}-run.yaml")

        if not config_file.exists():
            self.parser.error(
                f"File {str(config_file)} does not exist.\n\nPlease run `llama stack build` to generate (and optionally edit) a run.yaml file"
            )

        if not config_file.is_file():
            self.parser.error(
                f"Config file must be a valid file path, '{config_file}’ is not a file: type={type(config_file)}"
            )

        logger.info(f"Using run configuration: {config_file}")

        try:
            config_dict = yaml.safe_load(config_file.read_text())
        except yaml.parser.ParserError as e:
            self.parser.error(f"failed to load config file '{config_file}':\n {e}")

        try:
            config = parse_and_maybe_upgrade_config(config_dict)
        except AttributeError as e:
            self.parser.error(f"failed to parse config file '{config_file}':\n {e}")

        image_type, image_name = self._get_image_type_and_name(args)

        # If neither image type nor image name is provided, assume the server should be run directly
        # using the current environment packages.
        if not image_type and not image_name:
            logger.info("No image type or image name provided. Assuming environment packages.")
            from llama_stack.distribution.server.server import main as server_main

            # Build the server args from the current args passed to the CLI
            server_args = argparse.Namespace()
            for arg in vars(args):
                # If this is a function, avoid passing it
                # "args" contains:
                # func=<bound method StackRun._run_stack_run_cmd of <llama_stack.cli.stack.run.StackRun object at 0x10484b010>>
                if callable(getattr(args, arg)):
                    continue
                setattr(server_args, arg, getattr(args, arg))

            # Run the server
            server_main(server_args)
        else:
            run_args = formulate_run_args(image_type, image_name, config, template_name)

            run_args.extend([str(config_file), str(args.port)])
            if args.disable_ipv6:
                run_args.append("--disable-ipv6")

            if args.env:
                for env_var in args.env:
                    if "=" not in env_var:
                        self.parser.error(f"Environment variable '{env_var}' must be in KEY=VALUE format")
                        return
                    key, value = env_var.split("=", 1)  # split on first = only
                    if not key:
                        self.parser.error(f"Environment variable '{env_var}' has empty key")
                        return
                    run_args.extend(["--env", f"{key}={value}"])

            if args.tls_keyfile and args.tls_certfile:
                run_args.extend(["--tls-keyfile", args.tls_keyfile, "--tls-certfile", args.tls_certfile])
            run_command(run_args)
