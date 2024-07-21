import argparse
import os
import textwrap

from pathlib import Path

from toolchain.cli.subcommand import Subcommand
from toolchain.utils import DEFAULT_DUMP_DIR


CONFIGS_BASE_DIR = os.path.join(DEFAULT_DUMP_DIR, "configs")


class InferenceConfigure(Subcommand):
    """Llama cli for configuring llama toolchain configs"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama inference configure",
            description="Configure llama toolchain inference configs",
            epilog=textwrap.dedent(
                """
                Example:
                    llama inference configure
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_inference_configure_cmd)

    def _add_arguments(self):
        pass

    def read_user_inputs(self):
        checkpoint_dir = input("Enter the checkpoint directory for the model (e.g., ~/.llama/checkpoints/Meta-Llama-3-8B/): ")
        model_parallel_size = input("Enter model parallel size (e.g., 1 for 8B / 8 for 70B and 405B): ")

        return checkpoint_dir, model_parallel_size

    def write_output_yaml(
        self, 
        checkpoint_dir, 
        model_parallel_size, 
        yaml_output_path
    ):
        yaml_content = textwrap.dedent(f"""
            model_inference_config:
                impl_type: "inline"
                inline_config:
                    checkpoint_type: "pytorch"
                    checkpoint_dir: {checkpoint_dir}/
                    tokenizer_path: {checkpoint_dir}/tokenizer.model
                    model_parallel_size: {model_parallel_size}
                    max_seq_len: 2048
                    max_batch_size: 1
            """)
        with open(yaml_output_path, 'w') as yaml_file:
            yaml_file.write(yaml_content.strip())

        print(f"YAML configuration has been written to {yaml_output_path}")

    def _run_inference_configure_cmd(self, args: argparse.Namespace) -> None:
        checkpoint_dir, model_parallel_size = self.read_user_inputs()
        checkpoint_dir = os.path.expanduser(checkpoint_dir)
        
        if not (
            checkpoint_dir.endswith("original") or
            checkpoint_dir.endswith("original/")
        ):
            checkpoint_dir = os.path.join(checkpoint_dir, "original")

        os.makedirs(CONFIGS_BASE_DIR, exist_ok=True)
        yaml_output_path = Path(CONFIGS_BASE_DIR) / "inference.yaml"

        self.write_output_yaml(
            checkpoint_dir,
            model_parallel_size,
            yaml_output_path,
        )
