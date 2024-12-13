# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.tools import Tool, Tools

from .config import MetaReferenceToolConfig


class MetaReferenceTool(Tools):
    def __init__(self, config: MetaReferenceToolConfig):
        self.config = config

    async def register_tool(self, tool: Tool):
        pass
