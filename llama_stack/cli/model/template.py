# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import textwrap

from termcolor import colored

from llama_stack.cli.subcommand import Subcommand


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

    def _prompt_type(self, value):
        from llama_models.llama3.api.datatypes import ToolPromptFormat

        try:
            return ToolPromptFormat(value.lower())
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{value} is not a valid ToolPromptFormat. Choose from {', '.join(t.value for t in ToolPromptFormat)}"
            ) from None

    def _add_arguments(self):
        self.parser.add_argument(
            "-m",
            "--model-family",
            type=str,
            default="llama3_1",
            help="Model Family (llama3_1, llama3_X, etc.)",
        )
        self.parser.add_argument(
            "--name",
            type=str,
            help="Usecase template name (system_message, user_message, assistant_message, tool_message)...",
            required=False,
        )
        self.parser.add_argument(
            "--format",
            type=str,
            help="ToolPromptFormat (json or function_tag). This flag is used to print the template in a specific formats.",
            required=False,
            default="json",
        )
        self.parser.add_argument(
            "--raw",
            action="store_true",
            help="If set to true, don't pretty-print into a table. Useful to copy-paste.",
        )

    def _run_model_template_cmd(self, args: argparse.Namespace) -> None:
        from llama_models.llama3.api.interface import (
            list_jinja_templates,
            render_jinja_template,
        )

        from llama_stack.cli.table import print_table

        if args.name:
            tool_prompt_format = self._prompt_type(args.format)
            template, tokens_info = render_jinja_template(args.name, tool_prompt_format)
            rendered = ""
            for tok, is_special in tokens_info:
                if is_special:
                    rendered += colored(tok, "yellow", attrs=["bold"])
                else:
                    rendered += tok

            if not args.raw:
                rendered = rendered.replace("\n", "↵\n")
                print_table(
                    [
                        (
                            "Name",
                            colored(template.template_name, "white", attrs=["bold"]),
                        ),
                        ("Template", rendered),
                        ("Notes", template.notes),
                    ],
                    separate_rows=True,
                )
            else:
                print("Template: ", template.template_name)
                print("=" * 40)
                print(rendered)
        else:
            templates = list_jinja_templates()
            headers = ["Role", "Template Name"]
            print_table(
                [(t.role, t.template_name) for t in templates],
                headers,
            )
