# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

import argparse
import textwrap

from llama_toolchain.cli.model.template import ModelTemplate
from llama_toolchain.cli.subcommand import Subcommand


class ModelParser(Subcommand):
    """Llama cli for model interface apis"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "model",
            prog="llama model",
            description="Describe llama model interfaces",
            epilog=textwrap.dedent(
                """
                Example:
                    llama model <subcommand> <options>
                """
            ),
        )

        subparsers = self.parser.add_subparsers(title="model_subcommands")

        # Add sub-commandsa
        # ModelDescribe.create(subparsers)
        ModelTemplate.create(subparsers)
