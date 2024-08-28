# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_toolchain.cli.subcommand import Subcommand


class StackCreate(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "create",
            prog="llama distribution create",
            description="create a Llama stack distribution",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_distribution_create_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--name",
            type=str,
            help="Name of the distribution to create",
            required=True,
        )
        # for each Api the user wants to support, we should
        # get the list of available providers, ask which one the user
        # wants to pick and then ask for their configuration.

    def _run_distribution_create_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.distribution.registry import resolve_distribution_spec

        dist = resolve_distribution_spec(args.name)
        if dist is not None:
            self.parser.error(f"Stack with name {args.name} already exists")
            return

        raise NotImplementedError()
