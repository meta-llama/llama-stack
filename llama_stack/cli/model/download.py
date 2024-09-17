# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand


class ModelDownload(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "download",
            prog="llama model download",
            description="Download a model from llama.meta.com or Hugging Face Hub",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        from llama_stack.cli.download import setup_download_parser

        setup_download_parser(self.parser)
