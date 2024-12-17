# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
from typing import Dict

from llama_stack.providers.inline.tool_runtime.meta_reference.tools.base import BaseTool
from pydantic import BaseModel

from .ipython_tool.code_execution import (
    CodeExecutionContext,
    CodeExecutionRequest,
    CodeExecutor,
)


class CodeInterpreterConfig(BaseModel):
    matplotlib_dump_dir: str = None


class CodeInterpreterTool(BaseTool):

    @classmethod
    def tool_id(cls) -> str:
        return "code_interpreter"

    @classmethod
    def get_provider_config_type(cls):
        return CodeInterpreterConfig

    async def execute(self, code: str) -> Dict:
        config = CodeInterpreterConfig(**self.config)

        ctx = CodeExecutionContext(
            matplotlib_dump_dir=config.matplotlib_dump_dir or tempfile.mkdtemp(),
        )
        executor = CodeExecutor(ctx)

        req = CodeExecutionRequest(scripts=[code])
        result = executor.execute(req)

        response = {"status": result["process_status"], "output": []}

        for out_type in ["stdout", "stderr"]:
            if result[out_type]:
                response["output"].append(
                    {"type": out_type, "content": result[out_type]}
                )

        return response
