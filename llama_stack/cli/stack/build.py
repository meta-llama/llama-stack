# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import argparse
import textwrap

from llama_stack.cli.subcommand import Subcommand


class StackBuild(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "build",
            prog="llama stack build",
            description="Build a Llama stack container",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_stack_build_command)

    def _add_arguments(self):
        self.parser.add_argument(
            "--config",
            type=str,
            default=None,
            help="Path to a config file to use for the build. You can find example configs in llama_stack/distributions/**/build.yaml. If this argument is not provided, you will be prompted to enter information interactively",
        )

        self.parser.add_argument(
            "--template",
            type=str,
            default=None,
            help="Name of the example template config to use for build. You may use `llama stack build --list-templates` to check out the available templates",
        )

        self.parser.add_argument(
            "--list-templates",
            action="store_true",
            default=False,
            help="Show the available templates for building a Llama Stack distribution",
        )

        self.parser.add_argument(
            "--image-type",
            type=str,
            help="Image Type to use for the build. This can be either conda or container or venv. If not specified, will use the image type from the template config.",
            choices=["conda", "container", "venv"],
            default="conda",
        )

        self.parser.add_argument(
            "--image-name",
            type=str,
            help=textwrap.dedent(
                """[for image-type=conda|venv] Name of the conda or virtual environment to use for
the build. If not specified, currently active Conda environment will be used if found.
            """
            ),
            default=None,
        )
        self.parser.add_argument(
            "--print-deps-only",
            default=False,
            action="store_true",
            help="Print the dependencies for the stack only, without building the stack",
        )

        self.parser.add_argument(
            "--run",
            action="store_true",
            default=False,
            help="Run the stack after building using the same image type, name, and other applicable arguments",
        )

    def _run_stack_build_command(self, args: argparse.Namespace) -> None:
        # always keep implementation completely silo-ed away from CLI so CLI
        # can be fast to load and reduces dependencies
        from ._build import run_stack_build_command

        return run_stack_build_command(args)
