import textwrap
import unittest 
from datetime import datetime 

from llama_models.llama3_1.api.datatypes import (
    BuiltinTool,
    InstructModel,
    UserMessage,
    StopReason,
    SystemMessage,
)

from llama_toolchain.inference.api.endpoints import (
    ChatCompletionRequest
)
from llama_toolchain.inference.api.config import (
    OllamaImplConfig
)
from llama_toolchain.inference.ollama import (
    OllamaInference
)


class OllamaInferenceTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        ollama_config = OllamaImplConfig(
            model="llama3.1",
            url="http://localhost:11434",
        )

        # setup ollama 
        self.inference = OllamaInference(ollama_config)
        await self.inference.initialize()

        current_date = datetime.now()
        formatted_date = current_date.strftime("%d %B %Y")
        self.system_prompt = SystemMessage(
            content=textwrap.dedent(f"""
                Environment: ipython
                Tools: brave_search

                Cutting Knowledge Date: December 2023
                Today Date:{formatted_date}

            """),
        )

        self.system_prompt_with_custom_tool = SystemMessage(
            content=textwrap.dedent("""
                Environment: ipython
                Tools: brave_search, wolfram_alpha, photogen

                Cutting Knowledge Date: December 2023
                Today Date: 30 July 2024


                You have access to the following functions:

                Use the function 'get_boiling_point' to 'Get the boiling point of a imaginary liquids (eg. polyjuice)'
                {"name": "get_boiling_point", "description": "Get the boiling point of a imaginary liquids (eg. polyjuice)", "parameters": {"liquid_name": {"param_type": "string", "description": "The name of the liquid", "required": true}, "celcius": {"param_type": "boolean", "description": "Whether to return the boiling point in Celcius", "required": false}}}


                Think very carefully before calling functions.
                If you choose to call a function ONLY reply in the following format with no prefix or suffix:

                <function=example_function_name>{"example_name": "example_value"}</function>

                Reminder:
                - If looking for real time information use relevant functions before falling back to brave_search
                - Function calls MUST follow the specified format, start with <function= and end with </function>
                - Required parameters MUST be specified
                - Only call one function at a time
                - Put the entire function call reply on one line

                """
            ),
        )

    async def asyncTearDown(self):
        await self.inference.shutdown()

    async def test_text(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                UserMessage(
                    content="What is the capital of France?",
                ),
            ],
            stream=False,
        )
        iterator = self.inference.chat_completion(request)
        async for r in iterator:
            response = r

        self.assertTrue("Paris" in response.completion_message.content)
        self.assertEquals(response.completion_message.stop_reason, StopReason.end_of_turn)

    async def test_tool_call(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                self.system_prompt,
                UserMessage(
                    content="Who is the current US President?",
                ),
            ],
            stream=False,
        )
        iterator = self.inference.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message
        
        self.assertEquals(completion_message.content, "")
        self.assertEquals(completion_message.stop_reason, StopReason.end_of_message)
        
        self.assertEquals(len(completion_message.tool_calls), 1, completion_message.tool_calls)
        self.assertEquals(completion_message.tool_calls[0].tool_name, BuiltinTool.brave_search)
        self.assertTrue(
            "president" in completion_message.tool_calls[0].arguments["query"].lower()
        )
        
    async def test_code_execution(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                self.system_prompt,
                UserMessage(
                    content="Write code to compute the 5th prime number",
                ),
            ],
            stream=False,
        )
        iterator = self.inference.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message

        self.assertEquals(completion_message.content, "")
        self.assertEquals(completion_message.stop_reason, StopReason.end_of_message)
        
        self.assertEquals(len(completion_message.tool_calls), 1, completion_message.tool_calls)
        self.assertEquals(completion_message.tool_calls[0].tool_name, BuiltinTool.code_interpreter)
        code = completion_message.tool_calls[0].arguments["code"]
        self.assertTrue("def " in code.lower(), code)

    async def test_custom_tool(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                self.system_prompt_with_custom_tool,
                UserMessage(
                    content="Use provided function to find the boiling point of polyjuice in fahrenheit?",
                ),
            ],
            stream=False,
        )
        iterator = self.inference.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message
        
        self.assertEqual(completion_message.content, "")
        self.assertEquals(completion_message.stop_reason, StopReason.end_of_turn)
        
        self.assertEquals(len(completion_message.tool_calls), 1, completion_message.tool_calls)
        self.assertEquals(completion_message.tool_calls[0].tool_name, "get_boiling_point")

        args = completion_message.tool_calls[0].arguments
        self.assertTrue(isinstance(args, dict))
        self.assertTrue(args["liquid_name"], "polyjuice")
