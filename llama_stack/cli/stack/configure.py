# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.utils.config_dirs import BUILDS_BASE_DIR
from llama_stack.distribution.datatypes import *  # noqa: F403


class StackConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama stack configure",
            description="configure a llama stack distribution",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_configure_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "config",
            type=str,
            help="Path to the build config file (e.g. ~/.llama/builds/<image_type>/<name>-build.yaml). For docker, this could also be the name of the docker image. ",
        )

        self.parser.add_argument(
            "--output-dir",
            type=str,
            help="Path to the output directory to store generated run.yaml config file. If not specified, will use ~/.llama/build/<image_type>/<name>-run.yaml",
        )

    def _run_stack_configure_cmd(self, args: argparse.Namespace) -> None:
        import json
        import os
        import subprocess
        from pathlib import Path

        import pkg_resources

        import yaml
        from termcolor import cprint

        from llama_stack.distribution.build import ImageType
        from llama_stack.distribution.utils.exec import run_with_pty

        docker_image = None

        build_config_file = Path(args.config)
        if build_config_file.exists():
            with open(build_config_file, "r") as f:
                build_config = BuildConfig(**yaml.safe_load(f))
                self._configure_llama_distribution(build_config, args.output_dir)
            return

        conda_dir = (
            Path(os.path.expanduser("~/.conda/envs")) / f"llamastack-{args.config}"
        )
        output = subprocess.check_output(["bash", "-c", "conda info --json"])
        conda_envs = json.loads(output.decode("utf-8"))["envs"]

        for x in conda_envs:
            if x.endswith(f"/llamastack-{args.config}"):
                conda_dir = Path(x)
                break

        build_config_file = Path(conda_dir) / f"{args.config}-build.yaml"
        if build_config_file.exists():
            with open(build_config_file, "r") as f:
                build_config = BuildConfig(**yaml.safe_load(f))

            cprint(f"Using {build_config_file}...", "green")
            self._configure_llama_distribution(build_config, args.output_dir)
            return

        docker_image = args.config
        builds_dir = BUILDS_BASE_DIR / ImageType.docker.value
        if args.output_dir:
            builds_dir = Path(output_dir)
        os.makedirs(builds_dir, exist_ok=True)

        script = pkg_resources.resource_filename(
            "llama_stack", "distribution/configure_container.sh"
        )
        script_args = [script, docker_image, str(builds_dir)]

        return_code = run_with_pty(script_args)
        if return_code != 0:
            self.parser.error(
                f"Failed to configure container {docker_image} with return code {return_code}. Please run `llama stack build` first. "
            )

    def _configure_llama_distribution(
        self,
        build_config: BuildConfig,
        output_dir: Optional[str] = None,
    ):
        import json
        import os
        from pathlib import Path

        import yaml
        from termcolor import cprint

        from llama_stack.distribution.configure import (
            configure_api_providers,
            parse_and_maybe_upgrade_config,
        )
        from llama_stack.distribution.utils.serialize import EnumEncoder

        builds_dir = BUILDS_BASE_DIR / build_config.image_type
        if output_dir:
            builds_dir = Path(output_dir)
        os.makedirs(builds_dir, exist_ok=True)
        image_name = build_config.name.replace("::", "-")
        run_config_file = builds_dir / f"{image_name}-run.yaml"

        if run_config_file.exists():
            cprint(
                f"Configuration already exists at `{str(run_config_file)}`. Will overwrite...",
                "yellow",
                attrs=["bold"],
            )
            config_dict = yaml.safe_load(run_config_file.read_text())
            config = parse_and_maybe_upgrade_config(config_dict)
        else:
            config = StackRunConfig(
                built_at=datetime.now(),
                image_name=image_name,
                apis=list(build_config.distribution_spec.providers.keys()),
                providers={},
            )

        config = configure_api_providers(config, build_config.distribution_spec)

        config.docker_image = (
            image_name if build_config.image_type == "docker" else None
        )
        config.conda_env = image_name if build_config.image_type == "conda" else None

        with open(run_config_file, "w") as f:
            to_write = json.loads(json.dumps(config.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        cprint(
            f"> YAML configuration has been written to `{run_config_file}`.",
            color="blue",
        )

        cprint(
            f"You can now run `llama stack run {image_name} --port PORT`",
            color="green",
        )
