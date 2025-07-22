# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.stack.utils import print_subcommand_description
from llama_stack.cli.subcommand import Subcommand

from .describe import ShieldDescribe
from .list import ShieldList
from .register import ShieldRegister
from .unregister import ShieldUnregister


class ShieldParser(Subcommand):
    """Parser for shield commands"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "shield",
            prog="llama shield",
            description="Manage safety shields",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="shield_subcommands")

        # Add shield sub-commands
        ShieldList.create(subparsers)
        ShieldRegister.create(subparsers)
        ShieldDescribe.create(subparsers)
        ShieldUnregister.create(subparsers)

        print_subcommand_description(self.parser, subparsers)
