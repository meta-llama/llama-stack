import argparse
import textwrap

from llama_toolchain.cli.subcommand import Subcommand

from llama_toolchain.inference.server import main as inference_server_init


class InferenceStart(Subcommand):
    """Llama Inference cli for starting inference server"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "start",
            prog="llama inference start",
            description="Start an inference server",
            epilog=textwrap.dedent(
                """
                Example:
                    llama inference start <options>
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_inference_start_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "--port",
            type=int,
            help="Port to run the server on. Defaults to 5000",
            default=5000,
        )
        self.parser.add_argument(
            "--disable-ipv6",
            action="store_true",
            help="Disable IPv6 support",
            default=False,
        )
        self.parser.add_argument(
            "--config",
            type=str,
            help="Path to config file",
            default="inference"
        )

    def _run_inference_start_cmd(self, args: argparse.Namespace) -> None:
        inference_server_init(
            config_path=args.config,
            port=args.port,
            disable_ipv6=args.disable_ipv6,
        )
