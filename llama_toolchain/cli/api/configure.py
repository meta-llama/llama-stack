# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json

from pathlib import Path

import yaml

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import BUILDS_BASE_DIR
from llama_toolchain.distribution.datatypes import *  # noqa: F403
from termcolor import cprint


class ApiConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama api configure",
            description="configure a llama stack API provider",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_api_configure_cmd)

    def _add_arguments(self):
        from llama_toolchain.distribution.distribution import stack_apis

        allowed_args = [a.name for a in stack_apis()]
        self.parser.add_argument(
            "api",
            choices=allowed_args,
            help="Stack API (one of: {})".format(", ".join(allowed_args)),
        )
        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the provider build to fully configure",
            required=True,
        )

    def _run_api_configure_cmd(self, args: argparse.Namespace) -> None:
        name = args.name
        if not name.endswith(".yaml"):
            name += ".yaml"
        config_file = BUILDS_BASE_DIR / args.api / name
        if not config_file.exists():
            self.parser.error(
                f"Could not find {config_file}. Please run `llama api build` first"
            )
            return

        configure_llama_provider(config_file)


def configure_llama_provider(config_file: Path) -> None:
    from llama_toolchain.common.prompt_for_config import prompt_for_config
    from llama_toolchain.common.serialize import EnumEncoder
    from llama_toolchain.distribution.distribution import api_providers
    from llama_toolchain.distribution.dynamic import instantiate_class_type

    with open(config_file, "r") as f:
        config = PackageConfig(**yaml.safe_load(f))

    all_providers = api_providers()

    provider_configs = {}
    for api, stub_config in config.providers.items():
        providers = all_providers[Api(api)]
        provider_id = stub_config["provider_id"]
        if provider_id not in providers:
            raise ValueError(
                f"Unknown provider `{provider_id}` is not available for API `{api}`"
            )

        provider_spec = providers[provider_id]
        cprint(
            f"Configuring API surface: {api} ({provider_id})", "white", attrs=["bold"]
        )
        config_type = instantiate_class_type(provider_spec.config_class)

        try:
            existing_provider_config = config_type(**stub_config)
        except KeyError:
            existing_provider_config = None

        provider_config = prompt_for_config(
            config_type,
            existing_provider_config,
        )
        print("")

        provider_configs[api] = {
            "provider_id": provider_id,
            **provider_config.dict(),
        }

    config.providers = provider_configs
    with open(config_file, "w") as fp:
        to_write = json.loads(json.dumps(config.dict(), cls=EnumEncoder))
        fp.write(yaml.dump(to_write, sort_keys=False))

    print(f"YAML configuration has been written to {config_file}")
