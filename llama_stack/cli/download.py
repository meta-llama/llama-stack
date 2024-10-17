# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import asyncio
import json
import os
import shutil
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List

import httpx
from pydantic import BaseModel

from termcolor import cprint

from llama_stack.cli.subcommand import Subcommand


class Download(Subcommand):
    """Llama cli for downloading llama toolchain assets"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "download",
            prog="llama download",
            description="Download a model from llama.meta.com or Hugging Face Hub",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        setup_download_parser(self.parser)


def setup_download_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=["meta", "huggingface"],
        default="meta",
    )
    parser.add_argument(
        "--model-id",
        required=False,
        help="See `llama model list` or `llama model list --show-all` for the list of available models",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        required=False,
        default=None,
        help="Hugging Face API token. Needed for gated models like llama2/3. Will also try to read environment variable `HF_TOKEN` as default.",
    )
    parser.add_argument(
        "--meta-url",
        type=str,
        required=False,
        help="For source=meta, URL obtained from llama.meta.com after accepting license terms",
    )
    parser.add_argument(
        "--ignore-patterns",
        type=str,
        required=False,
        default="*.safetensors",
        help="""
For source=huggingface, files matching any of the patterns are not downloaded. Defaults to ignoring
safetensors files to avoid downloading duplicate weights.
""",
    )
    parser.add_argument(
        "--manifest-file",
        type=str,
        help="For source=meta, you can download models from a manifest file containing a file => URL mapping",
        required=False,
    )
    parser.set_defaults(func=partial(run_download_cmd, parser=parser))


def _hf_download(
    model: "Model",
    hf_token: str,
    ignore_patterns: str,
    parser: argparse.ArgumentParser,
):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    from llama_stack.distribution.utils.model_utils import model_local_dir

    repo_id = model.huggingface_repo
    if repo_id is None:
        raise ValueError(f"No repo id found for model {model.descriptor()}")

    output_dir = model_local_dir(model.descriptor())
    os.makedirs(output_dir, exist_ok=True)
    try:
        true_output_dir = snapshot_download(
            repo_id,
            local_dir=output_dir,
            ignore_patterns=ignore_patterns,
            token=hf_token,
            library_name="llama-stack",
        )
    except GatedRepoError:
        parser.error(
            "It looks like you are trying to access a gated repository. Please ensure you "
            "have access to the repository and have provided the proper Hugging Face API token "
            "using the option `--hf-token` or by running `huggingface-cli login`."
            "You can find your token by visiting https://huggingface.co/settings/tokens"
        )
    except RepositoryNotFoundError:
        parser.error(f"Repository '{repo_id}' not found on the Hugging Face Hub.")
    except Exception as e:
        parser.error(e)

    print(f"\nSuccessfully downloaded model to {true_output_dir}")


def _meta_download(model: "Model", meta_url: str, info: "LlamaDownloadInfo"):
    from llama_stack.distribution.utils.model_utils import model_local_dir

    output_dir = Path(model_local_dir(model.descriptor()))
    os.makedirs(output_dir, exist_ok=True)

    # I believe we can use some concurrency here if needed but not sure it is worth it
    for f in info.files:
        output_file = str(output_dir / f)
        url = meta_url.replace("*", f"{info.folder}/{f}")
        total_size = info.pth_size if "consolidated" in f else 0
        cprint(f"Downloading `{f}`...", "white")
        downloader = ResumableDownloader(url, output_file, total_size)
        asyncio.run(downloader.download())

    print(f"\nSuccessfully downloaded model to {output_dir}")
    cprint(f"\nMD5 Checksums are at: {output_dir / 'checklist.chk'}", "white")


def run_download_cmd(args: argparse.Namespace, parser: argparse.ArgumentParser):
    from llama_models.sku_list import llama_meta_net_info, resolve_model

    from .model.safety_models import prompt_guard_download_info, prompt_guard_model_sku

    if args.manifest_file:
        _download_from_manifest(args.manifest_file)
        return

    if args.model_id is None:
        parser.error("Please provide a model id")
        return

    # Check if model_id is a comma-separated list
    model_ids = [model_id.strip() for model_id in args.model_id.split(",")]

    prompt_guard = prompt_guard_model_sku()
    for model_id in model_ids:
        if model_id == prompt_guard.model_id:
            model = prompt_guard
            info = prompt_guard_download_info()
        else:
            model = resolve_model(model_id)
            if model is None:
                parser.error(f"Model {model_id} not found")
                continue
            info = llama_meta_net_info(model)

        if args.source == "huggingface":
            _hf_download(model, args.hf_token, args.ignore_patterns, parser)
        else:
            meta_url = args.meta_url or input(
                f"Please provide the signed URL for model {model_id} you received via email after visiting https://www.llama.com/llama-downloads/ (e.g., https://llama3-1.llamameta.net/*?Policy...): "
            )
            assert "llamameta.net" in meta_url
            _meta_download(model, meta_url, info)


class ModelEntry(BaseModel):
    model_id: str
    files: Dict[str, str]

    class Config:
        protected_namespaces = ()


class Manifest(BaseModel):
    models: List[ModelEntry]
    expires_on: datetime


def _download_from_manifest(manifest_file: str):
    from llama_stack.distribution.utils.model_utils import model_local_dir

    with open(manifest_file, "r") as f:
        d = json.load(f)
        manifest = Manifest(**d)

    if datetime.now() > manifest.expires_on:
        raise ValueError(f"Manifest URLs have expired on {manifest.expires_on}")

    for entry in manifest.models:
        print(f"Downloading model {entry.model_id}...")
        output_dir = Path(model_local_dir(entry.model_id))
        os.makedirs(output_dir, exist_ok=True)

        if any(output_dir.iterdir()):
            cprint(f"Output directory {output_dir} is not empty.", "red")

            while True:
                resp = input(
                    "Do you want to (C)ontinue download or (R)estart completely? (continue/restart): "
                )
                if resp.lower() == "restart" or resp.lower() == "r":
                    shutil.rmtree(output_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    break
                elif resp.lower() == "continue" or resp.lower() == "c":
                    print("Continuing download...")
                    break
                else:
                    cprint("Invalid response. Please try again.", "red")

        for fname, url in entry.files.items():
            output_file = str(output_dir / fname)
            downloader = ResumableDownloader(url, output_file)
            asyncio.run(downloader.download())


class ResumableDownloader:
    def __init__(
        self,
        url: str,
        output_file: str,
        total_size: int = 0,
        buffer_size: int = 32 * 1024,
    ):
        self.url = url
        self.output_file = output_file
        self.buffer_size = buffer_size
        self.total_size = total_size
        self.downloaded_size = 0
        self.start_size = 0
        self.start_time = 0

    async def get_file_info(self, client: httpx.AsyncClient) -> None:
        if self.total_size > 0:
            return

        # Force disable compression when trying to retrieve file size
        response = await client.head(
            self.url, follow_redirects=True, headers={"Accept-Encoding": "identity"}
        )
        response.raise_for_status()
        self.url = str(response.url)  # Update URL in case of redirects
        self.total_size = int(response.headers.get("Content-Length", 0))
        if self.total_size == 0:
            raise ValueError(
                "Unable to determine file size. The server might not support range requests."
            )

    async def download(self) -> None:
        self.start_time = time.time()
        async with httpx.AsyncClient(follow_redirects=True) as client:
            await self.get_file_info(client)

            if os.path.exists(self.output_file):
                self.downloaded_size = os.path.getsize(self.output_file)
                self.start_size = self.downloaded_size
                if self.downloaded_size >= self.total_size:
                    print(f"Already downloaded `{self.output_file}`, skipping...")
                    return

            additional_size = self.total_size - self.downloaded_size
            if not self.has_disk_space(additional_size):
                M = 1024 * 1024  # noqa
                print(
                    f"Not enough disk space to download `{self.output_file}`. "
                    f"Required: {(additional_size // M):.2f} MB"
                )
                raise ValueError(
                    f"Not enough disk space to download `{self.output_file}`"
                )

            while True:
                if self.downloaded_size >= self.total_size:
                    break

                # Cloudfront has a max-size limit
                max_chunk_size = 27_000_000_000
                request_size = min(
                    self.total_size - self.downloaded_size, max_chunk_size
                )
                headers = {
                    "Range": f"bytes={self.downloaded_size}-{self.downloaded_size + request_size}"
                }
                print(f"Downloading `{self.output_file}`....{headers}")
                try:
                    async with client.stream(
                        "GET", self.url, headers=headers
                    ) as response:
                        response.raise_for_status()
                        with open(self.output_file, "ab") as file:
                            async for chunk in response.aiter_bytes(self.buffer_size):
                                file.write(chunk)
                                self.downloaded_size += len(chunk)
                                self.print_progress()
                except httpx.HTTPError as e:
                    print(f"\nDownload interrupted: {e}")
                    print("You can resume the download by running the script again.")
                except Exception as e:
                    print(f"\nAn error occurred: {e}")

            print(f"\nFinished downloading `{self.output_file}`....")

    def print_progress(self) -> None:
        percent = (self.downloaded_size / self.total_size) * 100
        bar_length = 50
        filled_length = int(bar_length * self.downloaded_size // self.total_size)
        bar = "█" * filled_length + "-" * (bar_length - filled_length)

        elapsed_time = time.time() - self.start_time
        M = 1024 * 1024  # noqa

        speed = (
            (self.downloaded_size - self.start_size) / (elapsed_time * M)
            if elapsed_time > 0
            else 0
        )
        print(
            f"\rProgress: |{bar}| {percent:.2f}% "
            f"({self.downloaded_size // M}/{self.total_size // M} MB) "
            f"Speed: {speed:.2f} MiB/s",
            end="",
            flush=True,
        )

    def has_disk_space(self, file_size: int) -> bool:
        dir_path = os.path.dirname(os.path.abspath(self.output_file))
        free_space = shutil.disk_usage(dir_path).free
        return free_space > file_size
