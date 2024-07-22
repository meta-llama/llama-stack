import argparse
import textwrap

from llama_toolchain.cli.subcommand import Subcommand
from llama_models.llama3_1.api.interface import render_jinja_template, list_jinja_templates


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
            "sku",
            type=str,
            help="Model SKU",
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
