# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import time

from llama_stack.cli.subcommand import Subcommand
from llama_stack.distribution.utils.config_dirs import DISTRIBS_BASE_DIR


class StackList(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "list",
            prog="llama stack list",
            description="List the built stacks",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_list_cmd)

    def _add_arguments(self):
        pass

    def _run_stack_list_cmd(self, args: argparse.Namespace) -> None:
        from llama_stack.cli.table import print_table

        headers = ["Stack(s)", "Modified Time"]

        rows = []
        for stack in os.listdir(DISTRIBS_BASE_DIR):
            abs_path = os.path.join(DISTRIBS_BASE_DIR, stack)
            modified_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(abs_path)))
            rows.append(
                [
                    stack,
                    modified_time,
                ]
            )

        print_table(
            rows,
            headers,
            separate_rows=True,
        )
