# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from .download import Download
from .model import ModelParser
from .stack import StackParser
from .verify_download import VerifyDownload


class LlamaCLIParser:
    """Defines CLI parser for Llama CLI"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="llama",
            description="Welcome to the Llama CLI",
            add_help=True,
            formatter_class=argparse.RawTextHelpFormatter,
        )

        # Default command is to print help
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="subcommands")

        # Add sub-commands
        ModelParser.create(subparsers)
        StackParser.create(subparsers)
        Download.create(subparsers)
        VerifyDownload.create(subparsers)

        self.print_subcommand_description(subparsers)

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()

    def run(self, args: argparse.Namespace) -> None:
        args.func(args)

    def print_subcommand_description(self, subparsers: argparse._SubParsersAction):
        description_text = ""
        for name, subcommand in subparsers.choices.items():
            description = subcommand.description
            description_text += f"  {name:<21} {description}\n"
        self.parser.epilog = description_text


def main():
    parser = LlamaCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
