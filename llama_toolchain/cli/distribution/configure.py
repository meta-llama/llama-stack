# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os

from pathlib import Path

import pkg_resources

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.distribution.registry import all_registered_distributions
from llama_toolchain.utils import LLAMA_STACK_CONFIG_DIR


CONFIGS_BASE_DIR = os.path.join(LLAMA_STACK_CONFIG_DIR, "configs")


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
        distribs = all_registered_distributions()
        self.parser.add_argument(
            "--name",
            type=str,
            help="Mame of the distribution to configure",
            default="local-source",
            choices=[d.name for d in distribs],
        )

    def read_user_inputs(self):
        checkpoint_dir = input(
            "Enter the checkpoint directory for the model (e.g., ~/.llama/checkpoints/Meta-Llama-3-8B/): "
        )
        model_parallel_size = input(
            "Enter model parallel size (e.g., 1 for 8B / 8 for 70B and 405B): "
        )
        assert model_parallel_size.isdigit() and int(model_parallel_size) in {
            1,
            8,
        }, "model parallel size must be 1 or 8"

        return checkpoint_dir, model_parallel_size

    def write_output_yaml(self, checkpoint_dir, model_parallel_size, yaml_output_path):
        default_conf_path = pkg_resources.resource_filename(
            "llama_toolchain", "data/default_distribution_config.yaml"
        )
        with open(default_conf_path, "r") as f:
            yaml_content = f.read()

        yaml_content = yaml_content.format(
            checkpoint_dir=checkpoint_dir,
            model_parallel_size=model_parallel_size,
        )

        with open(yaml_output_path, "w") as yaml_file:
            yaml_file.write(yaml_content.strip())

        print(f"YAML configuration has been written to {yaml_output_path}")

    def _run_distribution_configure_cmd(self, args: argparse.Namespace) -> None:
        checkpoint_dir, model_parallel_size = self.read_user_inputs()
        checkpoint_dir = os.path.expanduser(checkpoint_dir)

        assert (
            Path(checkpoint_dir).exists() and Path(checkpoint_dir).is_dir()
        ), f"{checkpoint_dir} does not exist or it not a directory"

        os.makedirs(CONFIGS_BASE_DIR, exist_ok=True)
        yaml_output_path = Path(CONFIGS_BASE_DIR) / "distribution.yaml"

        self.write_output_yaml(
            checkpoint_dir,
            model_parallel_size,
            yaml_output_path,
        )
