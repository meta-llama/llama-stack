# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unittest

from llama_models.llama3.api import *  # noqa: F403
from llama_stack.apis.inference.inference import *  # noqa: F403
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_messages,
)

MODEL = "Llama3.1-8B-Instruct"


class PrepareMessagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_default(self):
        content = "Hello !"
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[
                UserMessage(content=content),
            ],
        )
        messages = chat_completion_request_to_messages(request)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1].content, content)
        self.assertTrue("Cutting Knowledge Date: December 2023" in messages[0].content)

    async def test_system_builtin_only(self):
        content = "Hello !"
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[
                UserMessage(content=content),
            ],
            tools=[
                ToolDefinition(tool_name=BuiltinTool.code_interpreter),
                ToolDefinition(tool_name=BuiltinTool.brave_search),
            ],
        )
        messages = chat_completion_request_to_messages(request)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1].content, content)
        self.assertTrue("Cutting Knowledge Date: December 2023" in messages[0].content)
        self.assertTrue("Tools: brave_search" in messages[0].content)

    async def test_system_custom_only(self):
        content = "Hello !"
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[
                UserMessage(content=content),
            ],
            tools=[
                ToolDefinition(
                    tool_name="custom1",
                    description="custom1 tool",
                    parameters={
                        "param1": ToolParamDefinition(
                            param_type="str",
                            description="param1 description",
                            required=True,
                        ),
                    },
                )
            ],
            tool_prompt_format=ToolPromptFormat.json,
        )
        messages = chat_completion_request_to_messages(request)
        self.assertEqual(len(messages), 3)
        self.assertTrue("Environment: ipython" in messages[0].content)

        self.assertTrue("Return function calls in JSON format" in messages[1].content)
        self.assertEqual(messages[-1].content, content)

    async def test_system_custom_and_builtin(self):
        content = "Hello !"
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[
                UserMessage(content=content),
            ],
            tools=[
                ToolDefinition(tool_name=BuiltinTool.code_interpreter),
                ToolDefinition(tool_name=BuiltinTool.brave_search),
                ToolDefinition(
                    tool_name="custom1",
                    description="custom1 tool",
                    parameters={
                        "param1": ToolParamDefinition(
                            param_type="str",
                            description="param1 description",
                            required=True,
                        ),
                    },
                ),
            ],
        )
        messages = chat_completion_request_to_messages(request)
        self.assertEqual(len(messages), 3)

        self.assertTrue("Environment: ipython" in messages[0].content)
        self.assertTrue("Tools: brave_search" in messages[0].content)

        self.assertTrue("Return function calls in JSON format" in messages[1].content)
        self.assertEqual(messages[-1].content, content)

    async def test_user_provided_system_message(self):
        content = "Hello !"
        system_prompt = "You are a pirate"
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=content),
            ],
            tools=[
                ToolDefinition(tool_name=BuiltinTool.code_interpreter),
            ],
        )
        messages = chat_completion_request_to_messages(request)
        self.assertEqual(len(messages), 2, messages)
        self.assertTrue(messages[0].content.endswith(system_prompt))

        self.assertEqual(messages[-1].content, content)
