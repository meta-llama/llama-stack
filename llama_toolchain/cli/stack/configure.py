# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
from pathlib import Path

import yaml
from termcolor import cprint

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import BUILDS_BASE_DIR
from llama_toolchain.core.datatypes import *  # noqa: F403


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
        from llama_toolchain.core.distribution_registry import (
            available_distribution_specs,
        )
        from llama_toolchain.core.package import BuildType

        allowed_ids = [d.distribution_id for d in available_distribution_specs()]
        self.parser.add_argument(
            "distribution",
            type=str,
            help='Distribution ("adhoc" or one of: {})'.format(allowed_ids),
        )
        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the build",
            required=True,
        )
        self.parser.add_argument(
            "--type",
            type=str,
            default="conda_env",
            choices=[v.value for v in BuildType],
        )

    def _run_stack_configure_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.core.package import BuildType

        build_type = BuildType(args.type)
        name = args.name
        config_file = (
            BUILDS_BASE_DIR
            / args.distribution
            / build_type.descriptor()
            / f"{name}.yaml"
        )
        if not config_file.exists():
            self.parser.error(
                f"Could not find {config_file}. Please run `llama stack build` first"
            )
            return

        configure_llama_distribution(config_file)


def configure_llama_distribution(config_file: Path) -> None:
    from llama_toolchain.common.serialize import EnumEncoder
    from llama_toolchain.core.configure import configure_api_providers

    with open(config_file, "r") as f:
        config = PackageConfig(**yaml.safe_load(f))

    if config.providers:
        cprint(
            f"Configuration already exists for {config.distribution_id}. Will overwrite...",
            "yellow",
            attrs=["bold"],
        )

    config.providers = configure_api_providers(config.providers)

    with open(config_file, "w") as fp:
        to_write = json.loads(json.dumps(config.dict(), cls=EnumEncoder))
        fp.write(yaml.dump(to_write, sort_keys=False))

    print(f"YAML configuration has been written to {config_file}")
