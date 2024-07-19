import argparse
import textwrap

from toolchain.cli.inference.start import InferenceStart
from toolchain.cli.subcommand import Subcommand


class InferenceParser(Subcommand):
    """Llama cli for inference apis"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "inference",
            prog="llama inference",
            description="Run inference on a llama model",
            epilog=textwrap.dedent(
                """
                Example:
                    llama inference start <options>
                """
            ),
        )

        subparsers = self.parser.add_subparsers(title="inference_subcommands")

        # Add sub-commandsa
        InferenceStart.create(subparsers)
