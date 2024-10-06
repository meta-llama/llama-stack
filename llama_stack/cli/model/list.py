# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_models.sku_list import all_registered_models

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table


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

    def _run_model_list_cmd(self, args: argparse.Namespace) -> None:
        from .safety_models import prompt_guard_model_sku

        headers = [
            "Model Descriptor",
            "Hugging Face Repo",
            "Context Length",
        ]

        rows = []
        for model in all_registered_models() + [prompt_guard_model_sku()]:
            if not args.show_all and not model.is_featured:
                continue

            descriptor = model.descriptor()
            rows.append(
                [
                    descriptor,
                    model.huggingface_repo,
                    f"{model.max_seq_length // 1024}K",
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )
