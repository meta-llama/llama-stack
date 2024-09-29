# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse

from llama_stack.cli.model.describe import ModelDescribe
from llama_stack.cli.model.download import ModelDownload
from llama_stack.cli.model.list import ModelList
from llama_stack.cli.model.prompt_format import ModelPromptFormat

from llama_stack.cli.subcommand import Subcommand


class ModelParser(Subcommand):
    """Llama cli for model interface apis"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "model",
            prog="llama model",
            description="Work with llama models",
        )

        subparsers = self.parser.add_subparsers(title="model_subcommands")

        # Add sub-commands
        ModelDownload.create(subparsers)
        ModelList.create(subparsers)
        ModelPromptFormat.create(subparsers)
        ModelDescribe.create(subparsers)
