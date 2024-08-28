# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_toolchain.cli.subcommand import Subcommand

from .configure import StackConfigure
from .create import StackCreate
from .install import StackInstall
from .list import StackList
from .start import StackStart


class StackParser(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "distribution",
            prog="llama distribution",
            description="Operate on llama stack distributions",
        )

        subparsers = self.parser.add_subparsers(title="distribution_subcommands")

        # Add sub-commands
        StackList.create(subparsers)
        StackInstall.create(subparsers)
        StackCreate.create(subparsers)
        StackConfigure.create(subparsers)
        StackStart.create(subparsers)
