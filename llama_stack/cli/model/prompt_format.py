# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import textwrap
from io import StringIO
from pathlib import Path

from llama_stack.cli.subcommand import Subcommand
from llama_stack.cli.table import print_table
from llama_stack.models.llama.datatypes import CoreModelId, ModelFamily, is_multimodal, model_family

ROOT_DIR = Path(__file__).parent.parent.parent


class ModelPromptFormat(Subcommand):
    """Llama model cli for describe a model prompt format (message formats)"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "prompt-format",
            prog="llama model prompt-format",
            description="Show llama model message formats",
            epilog=textwrap.dedent(
                """
                Example:
                    llama model prompt-format <options>
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_model_template_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "-m",
            "--model-name",
            type=str,
            help="Example: Llama3.1-8B or Llama3.2-11B-Vision, etc\n"
            "(Run `llama model list` to see a list of valid model names)",
        )
        self.parser.add_argument(
            "-l",
            "--list",
            action="store_true",
            help="List all available models",
        )

    def _run_model_template_cmd(self, args: argparse.Namespace) -> None:
        import importlib.resources

        # Only Llama 3.1 and 3.2 are supported
        supported_model_ids = [
            m for m in CoreModelId if model_family(m) in {ModelFamily.llama3_1, ModelFamily.llama3_2}
        ]

        model_list = [m.value for m in supported_model_ids]

        if args.list:
            headers = ["Model(s)"]
            rows = []
            for m in model_list:
                rows.append(
                    [
                        m,
                    ]
                )
            print_table(
                rows,
                headers,
                separate_rows=True,
            )
            return

        try:
            model_id = CoreModelId(args.model_name)
        except ValueError:
            self.parser.error(
                f"{args.model_name} is not a valid Model. Choose one from the list of valid models. "
                f"Run `llama model list` to see the valid model names."
            )

        if model_id not in supported_model_ids:
            self.parser.error(
                f"{model_id} is not a valid Model. Choose one from the list of valid models. "
                f"Run `llama model list` to see the valid model names."
            )

        llama_3_1_file = ROOT_DIR / "models" / "llama" / "llama3_1" / "prompt_format.md"
        llama_3_2_text_file = ROOT_DIR / "models" / "llama" / "llama3_2" / "text_prompt_format.md"
        llama_3_2_vision_file = ROOT_DIR / "models" / "llama" / "llama3_2" / "vision_prompt_format.md"
        if model_family(model_id) == ModelFamily.llama3_1:
            with importlib.resources.as_file(llama_3_1_file) as f:
                content = f.open("r").read()
        elif model_family(model_id) == ModelFamily.llama3_2:
            if is_multimodal(model_id):
                with importlib.resources.as_file(llama_3_2_vision_file) as f:
                    content = f.open("r").read()
            else:
                with importlib.resources.as_file(llama_3_2_text_file) as f:
                    content = f.open("r").read()

        render_markdown_to_pager(content)


def render_markdown_to_pager(markdown_content: str):
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.style import Style
    from rich.text import Text

    class LeftAlignedHeaderMarkdown(Markdown):
        def parse_header(self, token):
            level = token.type.count("h")
            content = Text(token.content)
            header_style = Style(color="bright_blue", bold=True)
            header = Text(f"{'#' * level} ", style=header_style) + content
            self.add_text(header)

    # Render the Markdown
    md = LeftAlignedHeaderMarkdown(markdown_content)

    # Capture the rendered output
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=100)  # Set a fixed width
    console.print(md)
    rendered_content = output.getvalue()
    print(rendered_content)
