# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.datatypes import *  # noqa: F403
from pathlib import Path

import yaml


class StackBuild(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "build",
            prog="llama stack build",
            description="Build a Llama stack container",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_build_command)

    def _add_arguments(self):
        self.parser.add_argument(
            "config",
            type=str,
            help="Path to a config file to use for the build. You may find example configs in llama_stack/distribution/example_configs",
        )

        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the llama stack build to override from template config",
        )

    def _run_stack_build_command_from_build_config(
        self, build_config: BuildConfig
    ) -> None:
        import json
        import os

        from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR
        from llama_stack.distribution.utils.serialize import EnumEncoder
        from llama_stack.distribution.build import ApiInput, build_image, ImageType
        from termcolor import cprint

        # save build.yaml spec for building same distribution again
        if build_config.image_type == ImageType.docker.value:
            # docker needs build file to be in the llama-stack repo dir to be able to copy over to the image
            llama_stack_path = Path(os.path.relpath(__file__)).parent.parent.parent
            build_dir = (
                llama_stack_path / "configs/distributions" / build_config.image_type
            )
        else:
            build_dir = DISTRIBS_BASE_DIR / build_config.image_type

        os.makedirs(build_dir, exist_ok=True)
        build_file_path = build_dir / f"{build_config.name}-build.yaml"

        with open(build_file_path, "w") as f:
            to_write = json.loads(json.dumps(build_config.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        build_image(build_config, build_file_path)

        cprint(
            f"Build spec configuration saved at {str(build_file_path)}",
            color="green",
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        from llama_stack.distribution.utils.prompt_for_config import prompt_for_config
        from llama_stack.distribution.utils.dynamic import instantiate_class_type

        if not args.config:
            self.parser.error(
                "No config file specified. Please use `llama stack build /path/to/*-build.yaml`. Example config files can be found in llama_stack/distribution/example_configs"
            )
            return

        with open(args.config, "r") as f:
            try:
                build_config = BuildConfig(**yaml.safe_load(f))
            except Exception as e:
                self.parser.error(f"Could not parse config file {args.config}: {e}")
                return
            if args.name:
                build_config.name = args.name
            self._run_stack_build_command_from_build_config(build_config)
