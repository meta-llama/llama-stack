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
        import os
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

        # if we get here, we need to try to find the conda build config file
        cprint(
            f"Could not find {build_config_file}. Trying conda build name instead...",
            color="green",
        )
        if os.getenv("CONDA_PREFIX"):
            conda_dir = (
                Path(os.getenv("CONDA_PREFIX")).parent / f"llamastack-{args.config}"
            )
            build_config_file = Path(conda_dir) / f"{args.config}-build.yaml"

            if build_config_file.exists():
                with open(build_config_file, "r") as f:
                    build_config = BuildConfig(**yaml.safe_load(f))

                self._configure_llama_distribution(build_config, args.output_dir)
                return
        
        # if we get here, we need to prompt user to try configure inside docker image
        self.parser.error(
            f"""
            Could not find {build_config_file}. Did you download a docker image?
            Try running `docker run -it --entrypoint "/bin/bash" <image_name>`
            `llama stack configure llamastack-build.yaml --output-dir  ./`
            to set a new run configuration file. 
            """,
        )
        return

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

        from llama_stack.distribution.configure import configure_api_providers
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
            config = StackRunConfig(**yaml.safe_load(run_config_file.read_text()))
        else:
            config = StackRunConfig(
                built_at=datetime.now(),
                image_name=image_name,
                apis_to_serve=[],
                api_providers={},
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

        if build_config.image_type == "conda":
            cprint(
                f"You can now run `llama stack run {image_name} --port PORT`",
                color="green",
            )
