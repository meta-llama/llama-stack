# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table
from llama_stack.models.llama.sku_list import all_registered_models


class ModelList(Subcommand):
    """List available llama models"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list",
            prog="llama model list",
            description="Show available llama models",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_model_list_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--show-all",
            action="store_true",
            help="Show all models (not just defaults)",
        )
        self.parser.add_argument(
            "-s",
            "--search",
            type=str,
            required=False,
            help="Search for the input string as a substring in the model descriptor(ID)",
        )

    def _run_model_list_cmd(self, args: argparse.Namespace) -> None:
        from .safety_models import prompt_guard_model_sku

        headers = [
            "Model Descriptor",
            "Model ID",
            "Context Length",
        ]

        rows = []
        for model in all_registered_models() + [prompt_guard_model_sku()]:
            if not args.show_all and not model.is_featured:
                continue

            descriptor = model.descriptor()
            if args.search:
                if args.search.lower() in descriptor.lower():
                    rows.append(
                        [
                            descriptor,
                            model.huggingface_repo,
                            f"{model.max_seq_length // 1024}K",
                        ]
                    )
            else:
                rows.append(
                    [
                        descriptor,
                        model.huggingface_repo,
                        f"{model.max_seq_length // 1024}K",
                    ]
                )
        if len(rows) == 0:
            print("Not found for search.")
        else:
            print_table(
                rows,
                headers,
                separate_rows=True,
            )
