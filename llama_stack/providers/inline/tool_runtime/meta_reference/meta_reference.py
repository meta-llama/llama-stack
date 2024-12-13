# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict

from llama_stack.apis.tools import InvokeToolResult, Tool, ToolRuntime
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from .config import MetaReferenceToolRuntimeConfig


class MetaReferenceToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime):
    def __init__(self, config: MetaReferenceToolRuntimeConfig):
        self.config = config

    async def initialize(self):
        pass

    async def register_tool(self, tool: Tool):
        print(f"registering tool {tool.identifier}")
        pass

    async def unregister_tool(self, tool_id: str) -> None:
        pass

    async def invoke_tool(self, tool_id: str, args: Dict[str, Any]) -> InvokeToolResult:
        pass
