# Run this test using the following command:
# python -m unittest tests/test_inference.py

import os
import unittest

from llama_models.llama3_1.api.datatypes import (
    InstructModel,
    UserMessage
)

from llama_toolchain.inference.api.config import (
    ImplType,
    InferenceConfig,
    InlineImplConfig,
    ModelCheckpointConfig,
    PytorchCheckpoint,
    CheckpointQuantizationFormat,
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

    async def asyncSetUp(self):
        # assert model exists on local
        model_dir = os.path.expanduser("~/.llama/checkpoints/Meta-Llama-3.1-8B-Instruct/original/")
        assert os.path.isdir(model_dir), HELPER_MSG

        tokenizer_path = os.path.join(model_dir, "tokenizer.model")
        assert os.path.exists(tokenizer_path), HELPER_MSG

        inference_config = InlineImplConfig(
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

        self.inference = InferenceImpl(inference_config)
        await self.inference.initialize()

    async def asyncTearDown(self):
        await self.inference.shutdown()

    async def test_inline_inference_no_streaming(self):
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

        async for chunk in iterator:
            response = chunk

        result = response.completion_message.content
        self.assertTrue("Paris" in result, result)

    async def test_inline_inference_streaming(self):
        request = ChatCompletionRequest(
            model=InstructModel.llama3_8b_chat,
            messages=[
                UserMessage(
                    content="What is the capital of France?",
                ),
            ],
            stream=True,
        )
        iterator = self.inference.chat_completion(request)

        events = []
        async for chunk in iterator:
            events.append(chunk.event)


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
