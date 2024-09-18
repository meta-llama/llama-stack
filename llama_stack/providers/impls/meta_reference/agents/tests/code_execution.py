# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unittest

from llama_models.llama3.api.datatypes import (
    Attachment,
    BuiltinTool,
    CompletionMessage,
    StopReason,
    ToolCall,
)

from ..tools.builtin import CodeInterpreterTool


class TestCodeInterpreter(unittest.IsolatedAsyncioTestCase):
    async def test_matplotlib(self):
        tool = CodeInterpreterTool()
        code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 1])
y = np.array([0, 10])

plt.plot(x, y)
plt.title('x = 1')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axvline(x=1, color='r')
plt.show()
        """
        message = CompletionMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    call_id="call_id",
                    tool_name=BuiltinTool.code_interpreter,
                    arguments={"code": code},
                )
            ],
            stop_reason=StopReason.end_of_message,
        )
        ret = await tool.run([message])

        self.assertEqual(len(ret), 1)

        output = ret[0].content
        self.assertIsInstance(output, Attachment)
        self.assertEqual(output.mime_type, "image/png")

    async def test_path_unlink(self):
        tool = CodeInterpreterTool()
        code = """
import os
from pathlib import Path
import tempfile

dpath = Path(os.environ["MPLCONFIGDIR"])
with open(dpath / "test", "w") as f:
    f.write("hello")

Path(dpath / "test").unlink()
print("_OK_")
        """
        message = CompletionMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    call_id="call_id",
                    tool_name=BuiltinTool.code_interpreter,
                    arguments={"code": code},
                )
            ],
            stop_reason=StopReason.end_of_message,
        )
        ret = await tool.run([message])

        self.assertEqual(len(ret), 1)

        output = ret[0].content
        self.assertTrue("_OK_" in output)


if __name__ == "__main__":
    unittest.main()
