# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import textwrap

from pathlib import Path

import pkg_resources

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.utils import DEFAULT_DUMP_DIR


CONFIGS_BASE_DIR = os.path.join(DEFAULT_DUMP_DIR, "configs")


class InferenceConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama inference configure",
            description="Configure llama toolchain inference configs",
            epilog=textwrap.dedent(
                """
                Example:
                    llama inference configure
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_inference_configure_cmd)

    def _add_arguments(self):
        pass

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
            "llama_toolchain", "data/default_inference_config.yaml"
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

    def _run_inference_configure_cmd(self, args: argparse.Namespace) -> None:
        checkpoint_dir, model_parallel_size = self.read_user_inputs()
        checkpoint_dir = os.path.expanduser(checkpoint_dir)

        assert (
            Path(checkpoint_dir).exists() and Path(checkpoint_dir).is_dir()
        ), f"{checkpoint_dir} does not exist or it not a directory"

        os.makedirs(CONFIGS_BASE_DIR, exist_ok=True)
        yaml_output_path = Path(CONFIGS_BASE_DIR) / "inference.yaml"

        self.write_output_yaml(
            checkpoint_dir,
            model_parallel_size,
            yaml_output_path,
        )
