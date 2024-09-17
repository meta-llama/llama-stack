# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand

from .build import StackBuild
from .configure import StackConfigure
from .list_apis import StackListApis
from .list_providers import StackListProviders
from .run import StackRun


class StackParser(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "stack",
            prog="llama stack",
            description="Operations for the Llama Stack / Distributions",
        )

        subparsers = self.parser.add_subparsers(title="stack_subcommands")

        # Add sub-commands
        StackBuild.create(subparsers)
        StackConfigure.create(subparsers)
        StackListApis.create(subparsers)
        StackListProviders.create(subparsers)
        StackRun.create(subparsers)
