# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import textwrap
from io import StringIO

from llama_models.datatypes import CoreModelId, is_multimodal, model_family, ModelFamily

from llama_stack.cli.subcommand import Subcommand


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
            default="llama3_1",
            help="Model Family (llama3_1, llama3_X, etc.)",
        )

    def _run_model_template_cmd(self, args: argparse.Namespace) -> None:
        import importlib.resources

        # Only Llama 3.1 and 3.2 are supported
        supported_model_ids = [
            m
            for m in CoreModelId
            if model_family(m) in {ModelFamily.llama3_1, ModelFamily.llama3_2}
        ]
        model_str = "\n".join([m.value for m in supported_model_ids])
        try:
            model_id = CoreModelId(args.model_name)
        except ValueError:
            self.parser.error(
                f"{args.model_name} is not a valid Model. Choose one from --\n{model_str}"
            )

        if model_id not in supported_model_ids:
            self.parser.error(
                f"{model_id} is not a valid Model. Choose one from --\n {model_str}"
            )

        llama_3_1_file = (
            importlib.resources.files("llama_models") / "llama3_1/prompt_format.md"
        )
        llama_3_2_text_file = (
            importlib.resources.files("llama_models") / "llama3_2/text_prompt_format.md"
        )
        llama_3_2_vision_file = (
            importlib.resources.files("llama_models")
            / "llama3_2/vision_prompt_format.md"
        )
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
