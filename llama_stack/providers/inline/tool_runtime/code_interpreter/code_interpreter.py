# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import tempfile
from typing import Any, Dict, List

from llama_stack.apis.tools import Tool, ToolGroupDef, ToolInvocationResult, ToolRuntime
from llama_stack.providers.datatypes import ToolsProtocolPrivate

from .code_execution import CodeExecutionContext, CodeExecutionRequest, CodeExecutor
from .config import CodeInterpreterToolConfig

log = logging.getLogger(__name__)


class CodeInterpreterToolRuntimeImpl(ToolsProtocolPrivate, ToolRuntime):
    def __init__(self, config: CodeInterpreterToolConfig):
        self.config = config
        ctx = CodeExecutionContext(
            matplotlib_dump_dir=tempfile.mkdtemp(),
        )
        self.code_executor = CodeExecutor(ctx)

    async def initialize(self):
        pass

    async def register_tool(self, tool: Tool):
        if tool.identifier != "code_interpreter":
            raise ValueError(f"Tool identifier {tool.identifier} is not supported")

    async def unregister_tool(self, tool_id: str) -> None:
        return

    async def discover_tools(self, tool_group: ToolGroupDef) -> List[Tool]:
        raise NotImplementedError("Code interpreter tool group not supported")

    async def invoke_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolInvocationResult:
        script = args["code"]
        req = CodeExecutionRequest(scripts=[script])
        res = self.code_executor.execute(req)
        pieces = [res["process_status"]]
        for out_type in ["stdout", "stderr"]:
            res_out = res[out_type]
            if res_out != "":
                pieces.extend([f"[{out_type}]", res_out, f"[/{out_type}]"])
                if out_type == "stderr":
                    log.error(f"ipython tool error: ↓\n{res_out}")
        return ToolInvocationResult(content="\n".join(pieces))
