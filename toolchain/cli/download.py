import argparse
import os
import textwrap
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

from toolchain.cli.subcommand import Subcommand


DEFAULT_OUTPUT_DIR = "/tmp/llama_toolchain_cache/"


class Download(Subcommand):
    """Llama cli for downloading llama toolchain assets"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "download",
            prog="llama download",
            description="Download a model from the Hugging Face Hub",
            epilog=textwrap.dedent(
                """\
            # Here are some examples on how to use this command:

            llama download --repo-id meta-llama/Llama-2-7b-hf --hf-token <HF_TOKEN>
            llama download --repo-id meta-llama/Llama-2-7b-hf --output-dir /data/my_custom_dir --hf-token <HF_TOKEN>
            HF_TOKEN=<HF_TOKEN> llama download --repo-id meta-llama/Llama-2-7b-hf

            The output directory will be used to load models and tokenizers for inference.
            """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_download_cmd)

    def _add_arguments(self):
        self.parser.add_argument(
            "repo_id",
            type=str,
            help="Name of the repository on Hugging Face Hub eg. llhf/Meta-Llama-3.1-70B-Instruct",
        )
        self.parser.add_argument(
            "--output-dir",
            type=Path,
            required=False,
            default=None,
            help=f"Directory in which to save the model. Defaults to `{DEFAULT_OUTPUT_DIR}<model_name>`.",
        )
        self.parser.add_argument(
            "--hf-token",
            type=str,
            required=False,
            default=os.getenv("HF_TOKEN", None),
            help="Hugging Face API token. Needed for gated models like Llama2. Will also try to read environment variable `HF_TOKEN` as default.",
        )

    def _run_download_cmd(self, args: argparse.Namespace):
        model_name = args.repo_id.split("/")[-1]

        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        output_dir = args.output_dir
        if output_dir is None:
            model_name = args.repo_id.split("/")[-1]
            output_dir = Path(DEFAULT_OUTPUT_DIR) / model_name

        try:
            true_output_dir = snapshot_download(
                args.repo_id,
                local_dir=output_dir,
                # "auto" will download to cache_dir and symlink files to local_dir
                # avoiding unnecessary duplicate copies
                local_dir_use_symlinks="auto",
                token=args.hf_token,
            )
        except GatedRepoError:
            self.parser.error(
                "It looks like you are trying to access a gated repository. Please ensure you "
                "have access to the repository and have provided the proper Hugging Face API token "
                "using the option `--hf-token` or by running `huggingface-cli login`."
                "You can find your token by visiting https://huggingface.co/settings/tokens"
            )
        except RepositoryNotFoundError:
            self.parser.error(
                f"Repository '{args.repo_id}' not found on the Hugging Face Hub."
            )
        except Exception as e:
            self.parser.error(e)

        print(f"Successfully downloaded model to {true_output_dir}")
