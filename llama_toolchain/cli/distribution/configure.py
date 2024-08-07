# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import shlex

from typing import Any, Dict

import yaml
from termcolor import cprint

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import DISTRIBS_BASE_DIR


class DistributionConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama distribution configure",
            description="configure a llama stack distribution",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_distribution_configure_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the distribution to configure",
            required=True,
        )

    def _run_distribution_configure_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.distribution.registry import resolve_distribution_spec

        config_file = DISTRIBS_BASE_DIR / args.name / "config.yaml"
        if not config_file.exists():
            self.parser.error(
                f"Could not find {config_file}. Please run `llama distribution install` first"
            )
            return

        # we need to find the spec from the name
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        dist = resolve_distribution_spec(config["spec"])
        if dist is None:
            raise ValueError(f"Could not find any registered spec `{config['spec']}`")

        configure_llama_distribution(dist, config)


def configure_llama_distribution(dist: "Distribution", config: Dict[str, Any]):
    from llama_toolchain.common.exec import run_command
    from llama_toolchain.common.prompt_for_config import prompt_for_config
    from llama_toolchain.common.serialize import EnumEncoder
    from llama_toolchain.distribution.dynamic import instantiate_class_type

    python_exe = run_command(shlex.split("which python"))
    # simple check
    conda_env = config["conda_env"]
    if conda_env not in python_exe:
        raise ValueError(
            f"Please re-run configure by activating the `{conda_env}` conda environment"
        )

    existing_config = config
    if "providers" in existing_config:
        cprint(
            f"Configuration already exists for {config['name']}. Will overwrite...",
            "yellow",
            attrs=["bold"],
        )

    provider_configs = {}
    for api, provider_spec in dist.provider_specs.items():
        cprint(f"Configuring API surface: {api.value}", "white", attrs=["bold"])
        config_type = instantiate_class_type(provider_spec.config_class)
        config = prompt_for_config(
            config_type,
            (
                config_type(**existing_config["providers"][api.value])
                if existing_config
                and "providers" in existing_config
                and api.value in existing_config["providers"]
                else None
            ),
        )
        provider_configs[api.value] = {
            "provider_id": provider_spec.provider_id,
            **config.dict(),
        }

    dist_config = {
        **existing_config,
        "providers": provider_configs,
    }

    config_path = DISTRIBS_BASE_DIR / existing_config["name"] / "config.yaml"
    with open(config_path, "w") as fp:
        dist_config = json.loads(json.dumps(dist_config, cls=EnumEncoder))
        fp.write(yaml.dump(dist_config, sort_keys=False))

    print(f"YAML configuration has been written to {config_path}")
