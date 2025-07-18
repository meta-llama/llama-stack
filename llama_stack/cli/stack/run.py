# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import subprocess
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
            nargs="?",  # Make it optional
            metavar="config | template",
            help="Path to config file to use for the run or name of known template (`llama stack list` for a list).",
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
            help="Name of the image to run.",
        )
        self.parser.add_argument(
            "--env",
            action="append",
            help="Environment variables to pass to the server in KEY=VALUE format. Can be specified multiple times.",
            metavar="KEY=VALUE",
        )
        self.parser.add_argument(
            "--image-type",
            type=str,
            help="Image Type used during the build. This can be either conda or container or venv.",
            choices=[e.value for e in ImageType if e.value != ImageType.CONTAINER.value],
        )
        self.parser.add_argument(
            "--enable-ui",
            action="store_true",
            help="Start the UI server",
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
        from llama_stack.distribution.utils.exec import formulate_run_args, run_command

        if args.enable_ui:
            self._start_ui_development_server(args.port)
        image_type, image_name = self._get_image_type_and_name(args)

        if args.config:
            try:
                from llama_stack.distribution.utils.config_resolution import Mode, resolve_config_or_template

                config_file = resolve_config_or_template(args.config, Mode.RUN)
            except ValueError as e:
                self.parser.error(str(e))
        else:
            config_file = None

        # Check if config is required based on image type
        if (image_type in [ImageType.CONDA.value, ImageType.VENV.value]) and not config_file:
            self.parser.error("Config file is required for venv and conda environments")

        if config_file:
            logger.info(f"Using run configuration: {config_file}")

            try:
                config_dict = yaml.safe_load(config_file.read_text())
            except yaml.parser.ParserError as e:
                self.parser.error(f"failed to load config file '{config_file}':\n {e}")

            try:
                config = parse_and_maybe_upgrade_config(config_dict)
                if not os.path.exists(str(config.external_providers_dir)):
                    os.makedirs(str(config.external_providers_dir), exist_ok=True)
            except AttributeError as e:
                self.parser.error(f"failed to parse config file '{config_file}':\n {e}")
        else:
            config = None

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
                if arg == "config":
                    server_args.config = str(config_file)
                else:
                    setattr(server_args, arg, getattr(args, arg))

            # Run the server
            server_main(server_args)
        else:
            run_args = formulate_run_args(image_type, image_name)

            run_args.extend([str(args.port)])

            if config_file:
                run_args.extend(["--config", str(config_file)])

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

            run_command(run_args)

    def _start_ui_development_server(self, stack_server_port: int):
        logger.info("Attempting to start UI development server...")
        # Check if npm is available
        npm_check = subprocess.run(["npm", "--version"], capture_output=True, text=True, check=False)
        if npm_check.returncode != 0:
            logger.warning(
                f"'npm' command not found or not executable. UI development server will not be started. Error: {npm_check.stderr}"
            )
            return

        ui_dir = REPO_ROOT / "llama_stack" / "ui"
        logs_dir = Path("~/.llama/ui/logs").expanduser()
        try:
            # Create logs directory if it doesn't exist
            logs_dir.mkdir(parents=True, exist_ok=True)

            ui_stdout_log_path = logs_dir / "stdout.log"
            ui_stderr_log_path = logs_dir / "stderr.log"

            # Open log files in append mode
            stdout_log_file = open(ui_stdout_log_path, "a")
            stderr_log_file = open(ui_stderr_log_path, "a")

            process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(ui_dir),
                stdout=stdout_log_file,
                stderr=stderr_log_file,
                env={**os.environ, "NEXT_PUBLIC_LLAMA_STACK_BASE_URL": f"http://localhost:{stack_server_port}"},
            )
            logger.info(f"UI development server process started in {ui_dir} with PID {process.pid}.")
            logger.info(f"Logs: stdout -> {ui_stdout_log_path}, stderr -> {ui_stderr_log_path}")
            logger.info(f"UI will be available at http://localhost:{os.getenv('LLAMA_STACK_UI_PORT', 8322)}")

        except FileNotFoundError:
            logger.error(
                "Failed to start UI development server: 'npm' command not found. Make sure npm is installed and in your PATH."
            )
        except Exception as e:
            logger.error(f"Failed to start UI development server in {ui_dir}: {e}")
