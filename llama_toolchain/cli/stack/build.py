# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

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
        from llama_toolchain.core.distribution_registry import available_distribution_specs
        from llama_toolchain.core.package import (
            BuildType,
        )

        allowed_ids = [d.distribution_id for d in available_distribution_specs()]
        self.parser.add_argument(
            "distribution",
            type=str,
            help="Distribution to build (either \"adhoc\" OR one of: {})".format(allowed_ids),
        )
        self.parser.add_argument(
            "api_providers",
            nargs='?',
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

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        from llama_toolchain.core.distribution_registry import resolve_distribution_spec
        from llama_toolchain.core.package import (
            ApiInput,
            BuildType,
            build_package,
        )

        api_inputs = []
        if args.distribution == "adhoc":
            if not args.api_providers:
                self.parser.error("You must specify API providers with (api=provider,...) for building an adhoc distribution")
                return

            parsed = parse_api_provider_tuples(args.api_providers, self.parser)
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
            docker_image = None
        else:
            if args.api_providers:
                self.parser.error("You cannot specify API providers for pre-registered distributions")
                return

            dist = resolve_distribution_spec(args.distribution)
            if dist is None:
                self.parser.error(f"Could not find distribution {args.distribution}")
                return

            for api, provider_id in dist.providers.items():
                api_inputs.append(
                    ApiInput(
                        api=api,
                        provider=provider_id,
                    )
                )
            docker_image = dist.docker_image

        build_package(
            api_inputs,
            build_type=BuildType(args.type),
            name=args.name,
            distribution_id=args.distribution,
            docker_image=docker_image,
        )
