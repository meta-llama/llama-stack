# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json

from llama_toolchain.cli.subcommand import Subcommand


class StackListDistributions(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list-distributions",
            prog="llama stack list-distributions",
            description="Show available Llama Stack Distributions",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_distribution_list_cmd)

    def _add_arguments(self):
        pass

    def _run_distribution_list_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.cli.table import print_table
        from llama_toolchain.core.distribution_registry import (
            available_distribution_specs,
        )

        # eventually, this should query a registry at llama.meta.com/llamastack/distributions
        headers = [
            "Distribution ID",
            "Providers",
            "Description",
        ]

        rows = []
        for spec in available_distribution_specs():
            providers = {k.value: v for k, v in spec.providers.items()}
            rows.append(
                [
                    spec.distribution_id,
                    json.dumps(providers, indent=2),
                    spec.description,
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )
