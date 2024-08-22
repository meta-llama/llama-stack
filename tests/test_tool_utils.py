import unittest

from llama_models.llama3.api import *  # noqa: F403
from llama_toolchain.inference.api import *  # noqa: F403
from llama_toolchain.inference.prepare_messages import prepare_messages_for_tools

MODEL = "Meta-Llama3.1-8B-Instruct"


class ToolUtilsTests(unittest.IsolatedAsyncioTestCase):
    async def test_system_default(self):
        content = "Hello !"
        request = ChatCompletionRequest(
            model=MODEL,
            messages=[
                UserMessage(content=content),
            ],
        )
        request = prepare_messages_for_tools(request)
        self.assertEqual(len(request.messages), 2)
        self.assertEqual(request.messages[-1].content, content)
        self.assertTrue(
            "Cutting Knowledge Date: December 2023" in request.messages[0].content
        )

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
        request = prepare_messages_for_tools(request)
        self.assertEqual(len(request.messages), 2)
        self.assertEqual(request.messages[-1].content, content)
        self.assertTrue(
            "Cutting Knowledge Date: December 2023" in request.messages[0].content
        )
        self.assertTrue("Tools: brave_search" in request.messages[0].content)

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
        request = prepare_messages_for_tools(request)
        self.assertEqual(len(request.messages), 3)
        self.assertTrue("Environment: ipython" in request.messages[0].content)

        self.assertTrue(
            "Return function calls in JSON format" in request.messages[1].content
        )
        self.assertEqual(request.messages[-1].content, content)

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
        request = prepare_messages_for_tools(request)
        self.assertEqual(len(request.messages), 3)

        self.assertTrue("Environment: ipython" in request.messages[0].content)
        self.assertTrue("Tools: brave_search" in request.messages[0].content)

        self.assertTrue(
            "Return function calls in JSON format" in request.messages[1].content
        )
        self.assertEqual(request.messages[-1].content, content)

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
        request = prepare_messages_for_tools(request)
        self.assertEqual(len(request.messages), 2, request.messages)
        self.assertTrue(request.messages[0].content.endswith(system_prompt))

        self.assertEqual(request.messages[-1].content, content)
