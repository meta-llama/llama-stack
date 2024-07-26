# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import re
import textwrap

from llama_models.llama3_1.api.interface import (
    list_jinja_templates,
    render_jinja_template,
)
from termcolor import colored, cprint

from llama_toolchain.cli.subcommand import Subcommand


def strip_ansi_colors(text):
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


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
            template, tokens_info = render_jinja_template(args.template)
            rendered = ""
            for tok, is_special in tokens_info:
                if is_special:
                    rendered += colored(tok, "yellow", attrs=["bold"])
                else:
                    rendered += tok
            rendered += "\n"
            print_table(
                [
                    ("Name", colored(template.template_name, "white", attrs=["bold"])),
                    ("Template", rendered),
                    ("Notes", template.notes),
                ],
                separate_rows=True,
            )
        else:
            templates = list_jinja_templates()
            headers = ["Role", "Template Name"]
            print_table(
                [(t.role, t.template_name) for t in templates],
                headers,
            )


def format_row(row, col_widths):
    def wrap(text, width):
        lines = []
        for line in text.split("\n"):
            if line.strip() == "":
                lines.append("")
            else:
                line = line.strip()
                lines.extend(
                    textwrap.wrap(
                        line, width, break_long_words=False, replace_whitespace=False
                    )
                )
        return lines

    wrapped = [wrap(item, width) for item, width in zip(row, col_widths)]
    max_lines = max(len(subrow) for subrow in wrapped)

    lines = []
    for i in range(max_lines):
        line = []
        for cell_lines, width in zip(wrapped, col_widths):
            value = cell_lines[i] if i < len(cell_lines) else ""
            line.append(value + " " * (width - len(strip_ansi_colors(value))))
        lines.append("| " + (" | ".join(line)) + " |")

    return "\n".join(lines)


def print_table(rows, headers=None, separate_rows: bool = False):
    if not headers:
        col_widths = [
            max(len(strip_ansi_colors(item)) for item in col) for col in zip(*rows)
        ]
    else:
        col_widths = [
            max(len(header), max(len(strip_ansi_colors(item)) for item in col))
            for header, col in zip(headers, zip(*rows))
        ]
    col_widths = [min(w, 80) for w in col_widths]

    header_line = "+".join("-" * (width + 2) for width in col_widths)
    header_line = f"+{header_line}+"

    if headers:
        print(header_line)
        cprint(format_row(headers, col_widths), "white", attrs=["bold"])

    print(header_line)
    for row in rows:
        print(format_row(row, col_widths))
        if separate_rows:
            print(header_line)

    if not separate_rows:
        print(header_line)
