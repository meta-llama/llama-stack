# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import unittest

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionMessage,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolConfig,
    UserMessage,
)
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    ToolDefinition,
    ToolParamDefinition,
    ToolPromptFormat,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_messages,
    chat_completion_request_to_prompt,
)

MODEL = "Llama3.1-8B-Instruct"
MODEL3_2 = "Llama3.2-3B-Instruct"


class PrepareMessagesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().set_debug(False)

    async def test_system_default(self):
        content = "Hello !"
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[
                UserMessage(content=content),
            ],
        )
        messages = chat_completion_request_to_messages(request, MODEL)
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
        messages = chat_completion_request_to_messages(request, MODEL)
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
            tool_config=ToolConfig(tool_prompt_format=ToolPromptFormat.json),
        )
        messages = chat_completion_request_to_messages(request, MODEL)
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
        messages = chat_completion_request_to_messages(request, MODEL)
        self.assertEqual(len(messages), 3)

        self.assertTrue("Environment: ipython" in messages[0].content)
        self.assertTrue("Tools: brave_search" in messages[0].content)

        self.assertTrue("Return function calls in JSON format" in messages[1].content)
        self.assertEqual(messages[-1].content, content)

    async def test_completion_message_encoding(self):
        request = ChatCompletionRequest(
            model=MODEL3_2,
            messages=[
                UserMessage(content="hello"),
                CompletionMessage(
                    content="",
                    stop_reason=StopReason.end_of_turn,
                    tool_calls=[
                        ToolCall(
                            tool_name="custom1",
                            arguments={"param1": "value1"},
                            call_id="123",
                        )
                    ],
                ),
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
                ),
            ],
            tool_config=ToolConfig(tool_prompt_format=ToolPromptFormat.python_list),
        )
        prompt = await chat_completion_request_to_prompt(request, request.model)
        self.assertIn('[custom1(param1="value1")]', prompt)

        request.model = MODEL
        request.tool_config.tool_prompt_format = ToolPromptFormat.json
        prompt = await chat_completion_request_to_prompt(request, request.model)
        self.assertIn(
            '{"type": "function", "name": "custom1", "parameters": {"param1": "value1"}}',
            prompt,
        )

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
        messages = chat_completion_request_to_messages(request, MODEL)
        self.assertEqual(len(messages), 2, messages)
        self.assertTrue(messages[0].content.endswith(system_prompt))

        self.assertEqual(messages[-1].content, content)

    async def test_repalce_system_message_behavior_builtin_tools(self):
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
            tool_config=ToolConfig(
                tool_choice="auto",
                tool_prompt_format="python_list",
                system_message_behavior="replace",
            ),
        )
        messages = chat_completion_request_to_messages(request, MODEL3_2)
        self.assertEqual(len(messages), 2, messages)
        self.assertTrue(messages[0].content.endswith(system_prompt))
        self.assertIn("Environment: ipython", messages[0].content)
        self.assertEqual(messages[-1].content, content)

    async def test_repalce_system_message_behavior_custom_tools(self):
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
            tool_config=ToolConfig(
                tool_choice="auto",
                tool_prompt_format="python_list",
                system_message_behavior="replace",
            ),
        )
        messages = chat_completion_request_to_messages(request, MODEL3_2)

        self.assertEqual(len(messages), 2, messages)
        self.assertTrue(messages[0].content.endswith(system_prompt))
        self.assertIn("Environment: ipython", messages[0].content)
        self.assertEqual(messages[-1].content, content)

    async def test_replace_system_message_behavior_custom_tools_with_template(self):
        content = "Hello !"
        system_prompt = "You are a pirate {{ function_description }}"
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[
                SystemMessage(content=system_prompt),
                UserMessage(content=content),
            ],
            tools=[
                ToolDefinition(tool_name=BuiltinTool.code_interpreter),
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
            tool_config=ToolConfig(
                tool_choice="auto",
                tool_prompt_format="python_list",
                system_message_behavior="replace",
            ),
        )
        messages = chat_completion_request_to_messages(request, MODEL3_2)

        self.assertEqual(len(messages), 2, messages)
        self.assertIn("Environment: ipython", messages[0].content)
        self.assertIn("You are a pirate", messages[0].content)
        # function description is present in the system prompt
        self.assertIn('"name": "custom1"', messages[0].content)
        self.assertEqual(messages[-1].content, content)
