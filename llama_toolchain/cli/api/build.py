# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from typing import Dict

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.distribution.datatypes import *  # noqa: F403


def parse_dependencies(
    dependencies: str, parser: argparse.ArgumentParser
) -> Dict[str, ProviderSpec]:
    from llama_toolchain.distribution.distribution import api_providers

    all_providers = api_providers()

    deps = {}
    for dep in dependencies.split(","):
        dep = dep.strip()
        if not dep:
            continue
        api_str, provider = dep.split("=")
        api = Api(api_str)

        provider = provider.strip()
        if provider not in all_providers[api]:
            parser.error(f"Provider `{provider}` is not available for API `{api}`")
            return
        deps[api] = all_providers[api][provider]

    return deps


class ApiBuild(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "build",
            prog="llama api build",
            description="Build a Llama stack API provider container",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_api_build_command)

    def _add_arguments(self):
        from llama_toolchain.distribution.distribution import stack_apis
        from llama_toolchain.distribution.package import (
            BuildType,
        )

        allowed_args = [a.name for a in stack_apis()]
        self.parser.add_argument(
            "api",
            choices=allowed_args,
            help="Stack API (one of: {})".format(", ".join(allowed_args)),
        )

        self.parser.add_argument(
            "--provider",
            type=str,
            help="The provider to package into the container",
            required=True,
        )
        self.parser.add_argument(
            "--dependencies",
            type=str,
            help="Comma separated list of (downstream_api=provider) dependencies needed for the API",
            required=False,
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

    def _run_api_build_command(self, args: argparse.Namespace) -> None:
        from llama_toolchain.distribution.package import (
            ApiInput,
            BuildType,
            build_package,
        )

        api_input = ApiInput(
            api=Api(args.api),
            provider=args.provider,
            dependencies=parse_dependencies(args.dependencies or "", self.parser),
        )

        build_package(
            [api_input],
            build_type=BuildType(args.type),
            name=args.name,
        )
