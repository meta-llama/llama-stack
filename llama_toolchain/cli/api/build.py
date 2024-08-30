# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from typing import Dict

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.core.datatypes import *  # noqa: F403


def parse_api_provider_tuples(
    tuples: str, parser: argparse.ArgumentParser
) -> Dict[str, ProviderSpec]:
    from llama_toolchain.core.distribution import api_providers

    all_providers = api_providers()

    deps = {}
    for dep in tuples.split(","):
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
        from llama_toolchain.core.package import (
            BuildType,
        )

        self.parser.add_argument(
            "api_providers",
            help="Comma separated list of (api=provider) tuples",
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
            default="conda_env",
            choices=[v.value for v in BuildType],
        )

    def _run_api_build_command(self, args: argparse.Namespace) -> None:
        from llama_toolchain.core.package import (
            ApiInput,
            BuildType,
            build_package,
        )

        parsed = parse_api_provider_tuples(args.api_providers, self.parser)
        api_inputs = []
        for api, provider_spec in parsed.items():
            for dep in provider_spec.api_dependencies:
                if dep not in parsed:
                    self.parser.error(f"API {api} needs dependency {dep} provided also")
                    return

            api_inputs.append(
                ApiInput(
                    api=api,
                    provider=provider_spec.provider_id,
                )
            )

        build_package(
            api_inputs,
            build_type=BuildType(args.type),
            name=args.name,
        )
