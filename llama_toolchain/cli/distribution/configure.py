# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import shlex

from pathlib import Path

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
        from llama_toolchain.distribution.registry import available_distributions

        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the distribution to configure",
            default="local-source",
            choices=[d.name for d in available_distributions()],
        )

    def _run_distribution_configure_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.distribution.registry import resolve_distribution

        dist = resolve_distribution(args.name)
        if dist is None:
            self.parser.error(f"Could not find distribution {args.name}")
            return

        env_file = DISTRIBS_BASE_DIR / dist.name / "conda.env"
        # read this file to get the conda env name
        assert env_file.exists(), f"Could not find conda env file {env_file}"
        with open(env_file, "r") as f:
            conda_env = f.read().strip()

        configure_llama_distribution(dist, conda_env)


def configure_llama_distribution(dist: "Distribution", conda_env: str):
    from llama_toolchain.common.exec import run_command
    from llama_toolchain.common.prompt_for_config import prompt_for_config
    from llama_toolchain.common.serialize import EnumEncoder
    from llama_toolchain.distribution.datatypes import PassthroughApiAdapter
    from llama_toolchain.distribution.dynamic import instantiate_class_type

    python_exe = run_command(shlex.split("which python"))
    # simple check
    if conda_env not in python_exe:
        raise ValueError(
            f"Please re-run configure by activating the `{conda_env}` conda environment"
        )

    existing_config = None
    config_path = Path(DISTRIBS_BASE_DIR) / dist.name / "config.yaml"
    if config_path.exists():
        cprint(
            f"Configuration already exists for {dist.name}. Will overwrite...",
            "yellow",
            attrs=["bold"],
        )
        with open(config_path, "r") as fp:
            existing_config = yaml.safe_load(fp)

    adapter_configs = {}
    for api_surface, adapter in dist.adapters.items():
        if isinstance(adapter, PassthroughApiAdapter):
            adapter_configs[api_surface.value] = adapter.dict()
        else:
            cprint(
                f"Configuring API surface: {api_surface.value}", "white", attrs=["bold"]
            )
            config_type = instantiate_class_type(adapter.config_class)
            config = prompt_for_config(
                config_type,
                (
                    config_type(**existing_config["adapters"][api_surface.value])
                    if existing_config
                    and api_surface.value in existing_config["adapters"]
                    else None
                ),
            )
            adapter_configs[api_surface.value] = {
                "adapter_id": adapter.adapter_id,
                **config.dict(),
            }

    dist_config = {
        "adapters": adapter_configs,
        "conda_env": conda_env,
    }

    with open(config_path, "w") as fp:
        dist_config = json.loads(json.dumps(dist_config, cls=EnumEncoder))
        fp.write(yaml.dump(dist_config, sort_keys=False))

    print(f"YAML configuration has been written to {config_path}")
