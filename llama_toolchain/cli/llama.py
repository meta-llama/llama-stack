import argparse

from llama_toolchain.cli.download import Download
from llama_toolchain.cli.inference.inference import InferenceParser


class LlamaCLIParser:
    """Defines CLI parser for Llama CLI"""

    def __init__(self):
        self.parser = argparse.ArgumentParser(
            prog="llama",
            description="Welcome to the LLama toolchain cli",
            add_help=True,
        )

        # Default command is to print help
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        subparsers = self.parser.add_subparsers(title="subcommands")

        # Add sub-commands
        Download.create(subparsers)
        InferenceParser.create(subparsers)

        # Import sub-commands from agentic_system if they exist
        try:
            from llama_agentic_system.cli.subcommand_modules import (
                SUBCOMMAND_MODULES,
            )
            for module in SUBCOMMAND_MODULES:
                module.create(subparsers)

        except ImportError:
            pass

    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()

    def run(self, args: argparse.Namespace) -> None:
        args.func(args)


def main():
    parser = LlamaCLIParser()
    args = parser.parse_args()
    parser.run(args)


if __name__ == "__main__":
    main()
