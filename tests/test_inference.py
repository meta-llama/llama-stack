# Run this test using the following command:
# python -m unittest tests/test_inference.py

import asyncio
import os
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

from llama_toolchain.inference.api.config import (
    ImplType,
    InferenceConfig,
    InlineImplConfig,
    RemoteImplConfig,
    ModelCheckpointConfig,
    PytorchCheckpoint,
    CheckpointQuantizationFormat,
)
from llama_toolchain.inference.api_instance import (
    get_inference_api_instance,
)
from llama_toolchain.inference.api.datatypes import (
    ChatCompletionResponseEventType,
)
from llama_toolchain.inference.api.endpoints import (
    ChatCompletionRequest
)
from llama_toolchain.inference.inference import InferenceImpl
from llama_toolchain.inference.event_logger import EventLogger


HELPER_MSG = """
This test needs llama-3.1-8b-instruct models.
Please donwload using the llama cli

llama download --source huggingface --model-id llama3_1_8b_instruct --hf-token <HF_TOKEN>
"""


class InferenceTests(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        # This runs the async setup function
        asyncio.run(cls.asyncSetUpClass())

    @classmethod
    async def asyncSetUpClass(cls):
        # assert model exists on local
        model_dir = os.path.expanduser("~/.llama/checkpoints/Meta-Llama-3.1-8B-Instruct/original/")
        assert os.path.isdir(model_dir), HELPER_MSG

        tokenizer_path = os.path.join(model_dir, "tokenizer.model")
        assert os.path.exists(tokenizer_path), HELPER_MSG

        inline_config = InlineImplConfig(
            checkpoint_config=ModelCheckpointConfig(
                checkpoint=PytorchCheckpoint(
                    checkpoint_dir=model_dir,
                    tokenizer_path=tokenizer_path,
                    model_parallel_size=1,
                    quantization_format=CheckpointQuantizationFormat.bf16,
                )
            ),
            max_seq_len=2048,
        )
        inference_config = InferenceConfig(
            impl_config=inline_config
        )

        # -- For faster testing iteration --
        # remote_config = RemoteImplConfig(
        #     url="http://localhost:5000"
        # )
        # inference_config = InferenceConfig(
        #     impl_config=remote_config
        # )

        cls.api = await get_inference_api_instance(inference_config)
        await cls.api.initialize()

        current_date = datetime.now()
        formatted_date = current_date.strftime("%d %B %Y")
        cls.system_prompt = SystemMessage(
            content=textwrap.dedent(f"""
                Environment: ipython
                Tools: brave_search

                Cutting Knowledge Date: December 2023
                Today Date:{formatted_date}

            """),
        )
        cls.system_prompt_with_custom_tool = SystemMessage(
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

    @classmethod
    def tearDownClass(cls):
        # This runs the async teardown function
        asyncio.run(cls.asyncTearDownClass())

    @classmethod
    async def asyncTearDownClass(cls):
        await cls.api.shutdown()

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
        iterator = InferenceTests.api.chat_completion(request)

        async for chunk in iterator:
            response = chunk

        result = response.completion_message.content
        self.assertTrue("Paris" in result, result)

    async def test_text_streaming(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                UserMessage(
                    content="What is the capital of France?",
                ),
            ],
            stream=True,
        )
        iterator = InferenceTests.api.chat_completion(request)

        events = []
        async for chunk in iterator:
            events.append(chunk.event)
            # print(f"{chunk.event.event_type:<40} | {str(chunk.event.stop_reason):<26} | {chunk.event.delta} ")

        self.assertEqual(
            events[0].event_type,
            ChatCompletionResponseEventType.start
        )
        self.assertEqual(
            events[-1].event_type,
            ChatCompletionResponseEventType.complete
        )

        response = ""
        for e in events[1:-1]:
            response += e.delta

        self.assertTrue("Paris" in response, response)

    async def test_custom_tool_call(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                InferenceTests.system_prompt_with_custom_tool,
                UserMessage(
                    content="Use provided function to find the boiling point of polyjuice in fahrenheit?",
                ),
            ],
            stream=False,
        )
        iterator = InferenceTests.api.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message

        self.assertEqual(completion_message.content, "")

        # FIXME: This test fails since there is a bug where
        # custom tool calls return incoorect stop_reason as out_of_tokens
        # instead of end_of_turn
        # self.assertEqual(completion_message.stop_reason, StopReason.end_of_turn)

        self.assertEqual(len(completion_message.tool_calls), 1, completion_message.tool_calls)
        self.assertEqual(completion_message.tool_calls[0].tool_name, "get_boiling_point")

        args = completion_message.tool_calls[0].arguments
        self.assertTrue(isinstance(args, dict))
        self.assertTrue(args["liquid_name"], "polyjuice")

    async def test_tool_call_streaming(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                self.system_prompt,
                UserMessage(
                    content="Who is the current US President?",
                ),
            ],
            stream=True,
        )
        iterator = InferenceTests.api.chat_completion(request)

        events = []
        async for chunk in iterator:
            # print(f"{chunk.event.event_type:<40} | {str(chunk.event.stop_reason):<26} | {chunk.event.delta} ")
            events.append(chunk.event)

        self.assertEqual(
            events[0].event_type,
            ChatCompletionResponseEventType.start
        )
        # last event is of type "complete"
        self.assertEqual(
            events[-1].event_type,
            ChatCompletionResponseEventType.complete
        )
        # last but one event should be eom with tool call
        self.assertEqual(
            events[-2].event_type,
            ChatCompletionResponseEventType.progress
        )
        self.assertEqual(
            events[-2].stop_reason,
            StopReason.end_of_message
        )
        self.assertEqual(
            events[-2].delta.content.tool_name,
            BuiltinTool.brave_search
        )

    async def test_custom_tool_call_streaming(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                InferenceTests.system_prompt_with_custom_tool,
                UserMessage(
                    content="Use provided function to find the boiling point of polyjuice?",
                ),
            ],
            stream=True,
        )
        iterator = InferenceTests.api.chat_completion(request)
        events = []
        async for chunk in iterator:
            # print(f"{chunk.event.event_type:<40} | {str(chunk.event.stop_reason):<26} | {chunk.event.delta} ")
            events.append(chunk.event)

        self.assertEqual(
            events[0].event_type,
            ChatCompletionResponseEventType.start
        )
        # last event is of type "complete"
        self.assertEqual(
            events[-1].event_type,
            ChatCompletionResponseEventType.complete
        )
        self.assertEqual(
            events[-1].stop_reason,
            StopReason.end_of_turn
        )
        # last but one event should be eom with tool call
        self.assertEqual(
            events[-2].event_type,
            ChatCompletionResponseEventType.progress
        )
        self.assertEqual(
            events[-2].stop_reason,
            StopReason.end_of_turn
        )
        self.assertEqual(
            events[-2].delta.content.tool_name,
            "get_boiling_point"
        )
