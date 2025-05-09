# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
from pathlib import Path
import shutil

from llama_stack.cli.subcommand import Subcommand


class StackRemove(Subcommand):
     """Remove the build stack"""

     def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "rm",
            prog="llama stack rm",
            description="Remove the build stack",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._remove_stack_command)

     def _add_arguments(self):
            self.parser.add_argument(
            "name",
            type=str,
            help="Name of the stack to delete",
      )

     def _remove_stack_command(self, args: argparse.Namespace) -> None:
            stack_dir = Path(args.name)
            if stack_dir.exists():
                shutil.rmtree(stack_dir)
                print(f"Successfully deleted stack: {args.name}")
            else:
                print(f"Stack not found: {args.name}")
