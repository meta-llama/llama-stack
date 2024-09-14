# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.core.datatypes import *  # noqa: F403
import yaml


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
        from llama_toolchain.core.distribution_registry import (
            available_distribution_specs,
        )
        from llama_toolchain.core.package import ImageType

        allowed_ids = [d.distribution_type for d in available_distribution_specs()]
        self.parser.add_argument(
            "--config",
            type=str,
            help="Path to a config file to use for the build",
        )

    def _run_stack_build_command_from_build_config(
        self, build_config: BuildConfig
    ) -> None:
        import json
        import os

        from llama_toolchain.common.config_dirs import DISTRIBS_BASE_DIR
        from llama_toolchain.common.serialize import EnumEncoder
        from llama_toolchain.core.distribution_registry import resolve_distribution_spec
        from llama_toolchain.core.package import ApiInput, build_package, ImageType
        from termcolor import cprint

        # expect build to take in a distribution spec file
        api_inputs = []
        for api, provider_type in build_config.distribution_spec.providers.items():
            api_inputs.append(
                ApiInput(
                    api=Api(api),
                    provider=provider_type,
                )
            )

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
                self._run_stack_build_command_from_build_config(build_config)
            return

        build_config = prompt_for_config(BuildConfig, None)
        self._run_stack_build_command_from_build_config(build_config)
