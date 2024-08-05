import textwrap
import unittest
from datetime import datetime

from llama_models.llama3_1.api.datatypes import (
    BuiltinTool,
    UserMessage,
    StopReason,
    SamplingParams,
    SamplingStrategy,
    SystemMessage,
)
from llama_toolchain.inference.api_instance import (
    get_inference_api_instance,
)
from llama_toolchain.inference.api.datatypes import (
    ChatCompletionResponseEventType,
)
from llama_toolchain.inference.api.endpoints import ChatCompletionRequest
from llama_toolchain.inference.api.config import InferenceConfig, OllamaImplConfig


class OllamaInferenceTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.valid_supported_model = "Meta-Llama3.1-8B-Instruct"
        ollama_config = OllamaImplConfig(
            model="llama3.1:8b-instruct-fp16",
            url="http://localhost:11434",
        )

        # setup ollama
        self.api = await get_inference_api_instance(
            InferenceConfig(impl_config=ollama_config)
        )
        await self.api.initialize()

        current_date = datetime.now()
        formatted_date = current_date.strftime("%d %B %Y")
        self.system_prompt = SystemMessage(
            content=textwrap.dedent(
                f"""
                Environment: ipython
                Tools: brave_search

                Cutting Knowledge Date: December 2023
                Today Date:{formatted_date}

            """
            ),
        )

        self.system_prompt_with_custom_tool = SystemMessage(
            content=textwrap.dedent(
                """
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
                - Put the entire function call reply on one line

                """
            ),
        )
        self.valid_supported_model = "Meta-Llama3.1-8B-Instruct"

    async def asyncTearDown(self):
        await self.api.shutdown()

    async def test_text(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                UserMessage(
                    content="What is the capital of France?",
                ),
            ],
            stream=False,
        )
        iterator = self.api.chat_completion(request)
        async for r in iterator:
            response = r

        self.assertTrue("Paris" in response.completion_message.content)
        self.assertEqual(
            response.completion_message.stop_reason, StopReason.end_of_turn
        )

    async def test_tool_call(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                self.system_prompt,
                UserMessage(
                    content="Who is the current US President?",
                ),
            ],
            stream=False,
        )
        iterator = self.api.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message

        self.assertEqual(completion_message.content, "")
        self.assertEqual(completion_message.stop_reason, StopReason.end_of_message)

        self.assertEqual(
            len(completion_message.tool_calls), 1, completion_message.tool_calls
        )
        self.assertEqual(
            completion_message.tool_calls[0].tool_name, BuiltinTool.brave_search
        )
        self.assertTrue(
            "president" in completion_message.tool_calls[0].arguments["query"].lower()
        )

    async def test_code_execution(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                self.system_prompt,
                UserMessage(
                    content="Write code to compute the 5th prime number",
                ),
            ],
            stream=False,
        )
        iterator = self.api.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message

        self.assertEqual(completion_message.content, "")
        self.assertEqual(completion_message.stop_reason, StopReason.end_of_message)

        self.assertEqual(
            len(completion_message.tool_calls), 1, completion_message.tool_calls
        )
        self.assertEqual(
            completion_message.tool_calls[0].tool_name, BuiltinTool.code_interpreter
        )
        code = completion_message.tool_calls[0].arguments["code"]
        self.assertTrue("def " in code.lower(), code)

    async def test_custom_tool(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                self.system_prompt_with_custom_tool,
                UserMessage(
                    content="Use provided function to find the boiling point of polyjuice?",
                ),
            ],
            stream=False,
        )
        iterator = self.api.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message

        self.assertEqual(completion_message.content, "")
        self.assertTrue(
            completion_message.stop_reason
            in {
                StopReason.end_of_turn,
                StopReason.end_of_message,
            }
        )

        self.assertEqual(
            len(completion_message.tool_calls), 1, completion_message.tool_calls
        )
        self.assertEqual(
            completion_message.tool_calls[0].tool_name, "get_boiling_point"
        )

        args = completion_message.tool_calls[0].arguments
        self.assertTrue(isinstance(args, dict))
        self.assertTrue(args["liquid_name"], "polyjuice")

    async def test_text_streaming(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                UserMessage(
                    content="What is the capital of France?",
                ),
            ],
            stream=True,
        )
        iterator = self.api.chat_completion(request)
        events = []
        async for chunk in iterator:
            # print(f"{chunk.event.event_type:<40} | {str(chunk.event.stop_reason):<26} | {chunk.event.delta} ")
            events.append(chunk.event)

        response = ""
        for e in events[1:-1]:
            response += e.delta

        self.assertEqual(events[0].event_type, ChatCompletionResponseEventType.start)
        # last event is of type "complete"
        self.assertEqual(
            events[-1].event_type, ChatCompletionResponseEventType.complete
        )
        # last but 1 event should be of type "progress"
        self.assertEqual(
            events[-2].event_type, ChatCompletionResponseEventType.progress
        )
        self.assertEqual(
            events[-2].stop_reason,
            None,
        )
        self.assertTrue("Paris" in response, response)

    async def test_tool_call_streaming(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                self.system_prompt,
                UserMessage(
                    content="Who is the current US President?",
                ),
            ],
            stream=True,
        )
        iterator = self.api.chat_completion(request)
        events = []
        async for chunk in iterator:
            # print(f"{chunk.event.event_type:<40} | {str(chunk.event.stop_reason):<26} | {chunk.event.delta} ")
            events.append(chunk.event)

        self.assertEqual(events[0].event_type, ChatCompletionResponseEventType.start)
        # last event is of type "complete"
        self.assertEqual(
            events[-1].event_type, ChatCompletionResponseEventType.complete
        )

    async def test_custom_tool_call_streaming(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                self.system_prompt_with_custom_tool,
                UserMessage(
                    content="Use provided function to find the boiling point of polyjuice?",
                ),
            ],
            stream=True,
        )
        iterator = self.api.chat_completion(request)
        events = []
        async for chunk in iterator:
            # print(f"{chunk.event.event_type:<40} | {str(chunk.event.stop_reason):<26} | {chunk.event.delta} ")
            events.append(chunk.event)

        self.assertEqual(events[0].event_type, ChatCompletionResponseEventType.start)
        # last event is of type "complete"
        self.assertEqual(
            events[-1].event_type, ChatCompletionResponseEventType.complete
        )
        self.assertEqual(events[-1].stop_reason, StopReason.end_of_turn)
        # last but one event should be eom with tool call
        self.assertEqual(
            events[-2].event_type, ChatCompletionResponseEventType.progress
        )
        self.assertEqual(events[-2].delta.content.tool_name, "get_boiling_point")
        self.assertEqual(events[-2].stop_reason, StopReason.end_of_turn)

    def test_resolve_ollama_model(self):
        ollama_model = self.api.resolve_ollama_model(self.valid_supported_model)
        self.assertEqual(ollama_model, "llama3.1:8b-instruct-fp16")

        invalid_model = "Meta-Llama3.1-8B"
        with self.assertRaisesRegex(
            AssertionError, f"Unsupported model: {invalid_model}"
        ):
            self.api.resolve_ollama_model(invalid_model)

    async def test_ollama_chat_options(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                UserMessage(
                    content="What is the capital of France?",
                ),
            ],
            stream=False,
            sampling_params=SamplingParams(
                sampling_strategy=SamplingStrategy.top_p,
                top_p=0.99,
                temperature=1.0,
            ),
        )
        options = self.api.get_ollama_chat_options(request)
        self.assertEqual(
            options,
            {
                "temperature": 1.0,
                "top_p": 0.99,
            },
        )
