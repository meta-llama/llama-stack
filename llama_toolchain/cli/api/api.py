# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_toolchain.cli.subcommand import Subcommand

from .build import ApiBuild


class ApiParser(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "api",
            prog="llama api",
            description="Operate on llama stack API providers",
        )

        subparsers = self.parser.add_subparsers(title="api_subcommands")

        # Add sub-commands
        ApiBuild.create(subparsers)
