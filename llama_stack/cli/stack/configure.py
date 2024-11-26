# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand


class StackConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama stack configure",
            description="configure a llama stack distribution",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_configure_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "config",
            type=str,
            help="Path to the build config file (e.g. ~/.llama/builds/<image_type>/<name>-build.yaml). For docker, this could also be the name of the docker image. ",
        )

        self.parser.add_argument(
            "--output-dir",
            type=str,
            help="Path to the output directory to store generated run.yaml config file. If not specified, will use ~/.llama/build/<image_type>/<name>-run.yaml",
        )

    def _run_stack_configure_cmd(self, args: argparse.Namespace) -> None:
        self.parser.error(
            """
            DEPRECATED! llama stack configure has been deprecated.
            Please use llama stack run <path/to/run.yaml> instead.
            Please see example run.yaml in /distributions folder.
            """
        )
