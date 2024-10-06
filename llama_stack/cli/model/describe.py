# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json

from llama_models.sku_list import resolve_model

from termcolor import colored

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table
from llama_stack.distribution.utils.serialize import EnumEncoder


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
            required=True,
        )

    def _run_model_describe_cmd(self, args: argparse.Namespace) -> None:
        from .safety_models import prompt_guard_model_sku

        prompt_guard = prompt_guard_model_sku()
        if args.model_id == prompt_guard.model_id:
            model = prompt_guard
        else:
            model = resolve_model(args.model_id)

        if model is None:
            self.parser.error(
                f"Model {args.model_id} not found; try 'llama model list' for a list of available models."
            )
            return

        rows = [
            (
                colored("Model", "white", attrs=["bold"]),
                colored(model.descriptor(), "white", attrs=["bold"]),
            ),
            ("Hugging Face ID", model.huggingface_repo or "<Not Available>"),
            ("Description", model.description),
            ("Context Length", f"{model.max_seq_length // 1024}K tokens"),
            ("Weights format", model.quantization_format.value),
            ("Model params.json", json.dumps(model.arch_args, indent=4)),
        ]

        if model.recommended_sampling_params is not None:
            sampling_params = model.recommended_sampling_params.dict()
            for k in ("max_tokens", "repetition_penalty"):
                del sampling_params[k]
            rows.append(
                (
                    "Recommended sampling params",
                    json.dumps(sampling_params, cls=EnumEncoder, indent=4),
                )
            )

        print_table(
            rows,
            separate_rows=True,
        )
