# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json

from llama_toolchain.cli.subcommand import Subcommand


class DistributionList(Subcommand):

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list",
            prog="llama distribution list",
            description="Show available llama stack distributions",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_distribution_list_cmd)

    def _add_arguments(self):
        pass

    def _run_distribution_list_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.cli.table import print_table
        from llama_toolchain.distribution.registry import available_distributions

        # eventually, this should query a registry at llama.meta.com/llamastack/distributions
        headers = [
            "Name",
            "Adapters",
            "Description",
        ]

        rows = []
        for dist in available_distributions():
            adapters = {k.value: v.adapter_id for k, v in dist.adapters.items()}
            rows.append(
                [
                    dist.name,
                    json.dumps(adapters, indent=2),
                    dist.description,
                ]
            )
        print_table(
            rows,
            headers,
            separate_rows=True,
        )
