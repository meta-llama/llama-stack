# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.subcommand import Subcommand


class ModelVerifyDownload(Subcommand):
    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "verify-download",
            prog="llama model verify-download",
            description="Verify the downloaded checkpoints' checksums",
            formatter_class=argparse.RawTextHelpFormatter,
        )

        from llama_stack.cli.verify_download import setup_verify_download_parser

        setup_verify_download_parser(self.parser)
