# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.core.datatypes import *  # noqa: F403
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
            "--config",
            type=str,
            help="Path to a config file to use for the build",
        )

        self.parser.add_argument(
            "--name",
            type=str,
            help="Override the name of the llama stack build",
        )

    def _run_stack_build_command_from_build_config(
        self, build_config: BuildConfig
    ) -> None:
        import json
        import os

        from llama_toolchain.common.config_dirs import DISTRIBS_BASE_DIR
        from llama_toolchain.common.serialize import EnumEncoder
        from llama_toolchain.core.package import ApiInput, build_package, ImageType
        from termcolor import cprint

        build_package(build_config)

        # save build.yaml spec for building same distribution again
        build_dir = DISTRIBS_BASE_DIR / build_config.image_type
        os.makedirs(build_dir, exist_ok=True)
        build_file_path = build_dir / f"{build_config.name}-build.yaml"

        with open(build_file_path, "w") as f:
            to_write = json.loads(json.dumps(build_config.dict(), cls=EnumEncoder))
            f.write(yaml.dump(to_write, sort_keys=False))

        cprint(
            f"Build spec configuration saved at {str(build_file_path)}",
            color="green",
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        from llama_toolchain.common.prompt_for_config import prompt_for_config
        from llama_toolchain.core.dynamic import instantiate_class_type

        if args.config:
            with open(args.config, "r") as f:
                try:
                    build_config = BuildConfig(**yaml.safe_load(f))
                except Exception as e:
                    self.parser.error(f"Could not parse config file {args.config}: {e}")
                    return
                build_config.name = args.name if args.name else build_config.name
                self._run_stack_build_command_from_build_config(build_config)
            return

        build_config = prompt_for_config(BuildConfig, build_config_default)
        self._run_stack_build_command_from_build_config(build_config)
