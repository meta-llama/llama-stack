# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import textwrap

from llama_models.llama3_1.api.interface import (
    list_jinja_templates,
    render_jinja_template,
)

from llama_toolchain.cli.subcommand import Subcommand


class ModelTemplate(Subcommand):
    """Llama model cli for describe a model template (message formats)"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "template",
            prog="llama model template",
            description="Show llama model message formats",
            epilog=textwrap.dedent(
                """
                Example:
                    llama model template <options>
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_model_template_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "-m",
            "--model-family",
            type=str,
            default="llama3_1",
            help="Model Family (llama3_1, llama3_X, etc.)",
        )
        self.parser.add_argument(
            "--template",
            type=str,
            help="Usecase template name (system_message, user_message, assistant_message, tool_message)...",
            required=False,
        )

    def _run_model_template_cmd(self, args: argparse.Namespace) -> None:
        if args.template:
            render_jinja_template(args.template)
        else:
            list_jinja_templates()
