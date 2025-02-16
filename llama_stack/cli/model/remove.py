# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import shutil
import sys

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.utils.config_dirs import DEFAULT_CHECKPOINT_DIR


def _ask_for_confirm(model) -> bool:
    input_text = input(f"Are you sure you want to remove {model}? (y/n): ").strip().lower()
    if input_text == "y":
        return True
    elif input_text == "n":
        return False
    return False


def _remove_model(model) -> None:
    model_path = os.path.join(DEFAULT_CHECKPOINT_DIR, model)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        print(f"{model} removed.")
    else:
        print(f"{model} does not exist.")
        sys.exit(1)


class ModelRemove(Subcommand):
    """Remove the downloaded llama model"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "remove",
            prog="llama model remove",
            description="Remove the downloaded llama model",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_model_remove_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "-m",
            "--model",
            required=True,
            help="Specify the llama downloaded model name",
        )
        self.parser.add_argument(
            "-f",
            "--force",
            action="store_true",
            help="Used to forcefully remove the llama model from the storage without further confirmation",
        )

    def _run_model_remove_cmd(self, args: argparse.Namespace) -> None:
        if args.force:
            _remove_model(args.model)
        else:
            confirm = _ask_for_confirm(args.model)
            if confirm:
                _remove_model(args.model)
            else:
                print("Removal aborted.")
