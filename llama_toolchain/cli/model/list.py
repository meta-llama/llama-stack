# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_models.llama3_1.api.sku_list import llama3_1_model_list

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.cli.table import print_table


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
            "-m",
            "--model-family",
            type=str,
            default="llama3_1",
            help="Model Family (llama3_1, llama3_X, etc.)",
        )

    def _run_model_list_cmd(self, args: argparse.Namespace) -> None:
        models = llama3_1_model_list()
        headers = [
            "Model ID",
            "HuggingFace ID",
            "Context Length",
            "Hardware Requirements",
        ]

        rows = []
        for model in models:
            req = model.hardware_requirements
            rows.append(
                [
                    model.sku.value,
                    model.huggingface_id,
                    f"{model.max_seq_length // 1024}K",
                    f"{req.gpu_count} GPU{'s' if req.gpu_count > 1 else ''}, each >= {req.memory_gb_per_gpu}GB VRAM",
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )
