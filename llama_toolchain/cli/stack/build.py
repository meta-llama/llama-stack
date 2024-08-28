# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from typing import Dict

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.distribution.datatypes import *  # noqa: F403


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
        from llama_toolchain.distribution.registry import available_distribution_specs
        from llama_toolchain.distribution.package import (
            BuildType,
        )

        allowed_ids = [d.distribution_id for d in available_distribution_specs()]
        self.parser.add_argument(
            "distribution",
            type=str,
            choices=allowed_ids,
            help="Distribution to build (one of: {})".format(allowed_ids),
        )

        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the build target (image, conda env)",
            required=True,
        )
        self.parser.add_argument(
            "--type",
            type=str,
            default="container",
            choices=[v.value for v in BuildType],
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        from llama_toolchain.distribution.registry import resolve_distribution_spec
        from llama_toolchain.distribution.package import (
            ApiInput,
            BuildType,
            build_package,
        )

        dist = resolve_distribution_spec(args.distribution)
        if dist is None:
            self.parser.error(f"Could not find distribution {args.distribution}")
            return

        api_inputs = []
        for api, provider_id in dist.providers.items():
            api_inputs.append(
                ApiInput(
                    api=api,
                    provider=provider_id,
                    dependencies={},
                )
            )

        build_package(
            api_inputs,
            build_type=BuildType(args.type),
            name=args.name,
            distribution_id=args.distribution,
            docker_image=dist.docker_image,
        )
