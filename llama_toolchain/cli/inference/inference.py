# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import textwrap

from llama_toolchain.cli.inference.configure import InferenceConfigure
from llama_toolchain.cli.subcommand import Subcommand


class InferenceParser(Subcommand):
    """Llama cli for inference apis"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "inference",
            prog="llama inference",
            description="Run inference on a llama model",
            epilog=textwrap.dedent(
                """
                Example:
                    llama inference start <options>
                """
            ),
        )

        subparsers = self.parser.add_subparsers(title="inference_subcommands")

        # Add sub-commands
        InferenceConfigure.create(subparsers)
