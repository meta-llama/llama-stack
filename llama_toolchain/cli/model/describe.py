# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json

from llama_models.llama3_1.api.sku_list import llama3_1_model_list

from termcolor import colored

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.cli.table import print_table


class ModelDescribe(Subcommand):
    """Show details about a model"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "describe",
            prog="llama model describe",
            description="Show details about a llama model",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_model_describe_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "-m",
            "--model-id",
            type=str,
        )

    def _run_model_describe_cmd(self, args: argparse.Namespace) -> None:
        models = llama3_1_model_list()
        by_id = {model.sku.value: model for model in models}

        if args.model_id not in by_id:
            print(
                f"Model {args.model_id} not found; try 'llama model list' for a list of available models."
            )
            return

        model = by_id[args.model_id]

        rows = [
            (
                colored("Model", "white", attrs=["bold"]),
                colored(model.sku.value, "white", attrs=["bold"]),
            ),
            ("HuggingFace ID", model.huggingface_id or "<Not Available>"),
            ("Description", model.description_markdown),
            ("Context Length", f"{model.max_seq_length // 1024}K tokens"),
            ("Weights format", model.quantization_format.value),
            ("Model params.json", json.dumps(model.model_args, indent=4)),
        ]

        print_table(
            rows,
            separate_rows=True,
        )
