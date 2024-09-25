# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unittest

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.inference.api import *  # noqa: F403
from llama_stack.inference.ollama.config import OllamaImplConfig
from llama_stack.inference.ollama.ollama import get_provider_impl


class OllamaInferenceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        ollama_config = OllamaImplConfig(url="http://localhost:11434")

        # setup ollama
        self.api = await get_provider_impl(ollama_config, {})
        await self.api.initialize()

        self.custom_tool_defn = ToolDefinition(
            tool_name="get_boiling_point",
            description="Get the boiling point of a imaginary liquids (eg. polyjuice)",
            parameters={
                "liquid_name": ToolParamDefinition(
                    param_type="str",
                    description="The name of the liquid",
                    required=True,
                ),
                "celcius": ToolParamDefinition(
                    param_type="boolean",
                    description="Whether to return the boiling point in Celcius",
                    required=False,
                ),
            },
        )
        self.valid_supported_model = "Llama3.1-8B-Instruct"

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
        iterator = self.api.chat_completion(
            request.model, request.messages, stream=request.stream
        )
        async for r in iterator:
            response = r
        print(response.completion_message.content)
        self.assertTrue("Paris" in response.completion_message.content)
        self.assertEqual(
            response.completion_message.stop_reason, StopReason.end_of_turn
        )

    async def test_tool_call(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                UserMessage(
                    content="Who is the current US President?",
                ),
            ],
            stream=False,
            tools=[ToolDefinition(tool_name=BuiltinTool.brave_search)],
        )
        iterator = self.api.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message

        self.assertEqual(completion_message.content, "")
        self.assertEqual(completion_message.stop_reason, StopReason.end_of_turn)

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
                UserMessage(
                    content="Write code to compute the 5th prime number",
                ),
            ],
            tools=[ToolDefinition(tool_name=BuiltinTool.code_interpreter)],
            stream=False,
        )
        iterator = self.api.chat_completion(request)
        async for r in iterator:
            response = r

        completion_message = response.completion_message

        self.assertEqual(completion_message.content, "")
        self.assertEqual(completion_message.stop_reason, StopReason.end_of_turn)

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
                UserMessage(
                    content="Use provided function to find the boiling point of polyjuice?",
                ),
            ],
            stream=False,
            tools=[self.custom_tool_defn],
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
                UserMessage(
                    content="Using web search tell me who is the current US President?",
                ),
            ],
            stream=True,
            tools=[ToolDefinition(tool_name=BuiltinTool.brave_search)],
        )
        iterator = self.api.chat_completion(request)
        events = []
        async for chunk in iterator:
            events.append(chunk.event)

        self.assertEqual(events[0].event_type, ChatCompletionResponseEventType.start)
        # last event is of type "complete"
        self.assertEqual(
            events[-1].event_type, ChatCompletionResponseEventType.complete
        )
        # last but one event should be eom with tool call
        self.assertEqual(
            events[-2].event_type, ChatCompletionResponseEventType.progress
        )
        self.assertEqual(events[-2].stop_reason, StopReason.end_of_turn)
        self.assertEqual(events[-2].delta.content.tool_name, BuiltinTool.brave_search)

    async def test_custom_tool_call_streaming(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                UserMessage(
                    content="Use provided function to find the boiling point of polyjuice?",
                ),
            ],
            stream=True,
            tools=[self.custom_tool_defn],
            tool_prompt_format=ToolPromptFormat.function_tag,
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

        invalid_model = "Llama3.1-8B"
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

    async def test_multi_turn(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                UserMessage(
                    content="Search the web and tell me who the "
                    "44th president of the United States was",
                ),
                ToolResponseMessage(
                    call_id="1",
                    tool_name=BuiltinTool.brave_search,
                    content='{"query": "44th president of the United States", "top_k": [{"title": "Barack Obama | The White House", "url": "https://www.whitehouse.gov/about-the-white-house/presidents/barack-obama/", "description": "<strong>Barack Obama</strong> served as the 44th President of the United States. His story is the American story \\u2014 values from the heartland, a middle-class upbringing in a strong family, hard work and education as the means of getting ahead, and the conviction that a life so blessed should be lived in service ...", "type": "search_result"}, {"title": "Barack Obama \\u2013 The White House", "url": "https://trumpwhitehouse.archives.gov/about-the-white-house/presidents/barack-obama/", "description": "After working his way through college with the help of scholarships and student loans, <strong>President Obama</strong> moved to Chicago, where he worked with a group of churches to help rebuild communities devastated by the closure of local steel plants.", "type": "search_result"}, [{"type": "video_result", "url": "https://www.instagram.com/reel/CzMZbJmObn9/", "title": "Fifteen years ago, on Nov. 4, Barack Obama was elected as ...", "description": ""}, {"type": "video_result", "url": "https://video.alexanderstreet.com/watch/the-44th-president-barack-obama?context=channel:barack-obama", "title": "The 44th President (Barack Obama) - Alexander Street, a ...", "description": "You need to enable JavaScript to run this app"}, {"type": "video_result", "url": "https://www.youtube.com/watch?v=iyL7_2-em5k", "title": "Barack Obama for Kids | Learn about the life and contributions ...", "description": "Enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on YouTube."}, {"type": "video_result", "url": "https://www.britannica.com/video/172743/overview-Barack-Obama", "title": "President of the United States of America Barack Obama | Britannica", "description": "[NARRATOR] Barack Obama was elected the 44th president of the United States in 2008, becoming the first African American to hold the office. Obama vowed to bring change to the political system."}, {"type": "video_result", "url": "https://www.youtube.com/watch?v=rvr2g8-5dcE", "title": "The 44th President: In His Own Words - Toughest Day | Special ...", "description": "President Obama reflects on his toughest day in the Presidency and seeing Secret Service cry for the first time. Watch the premiere of The 44th President: In..."}]]}',
                ),
            ],
            stream=True,
            tools=[ToolDefinition(tool_name=BuiltinTool.brave_search)],
        )
        iterator = self.api.chat_completion(request)

        events = []
        async for chunk in iterator:
            events.append(chunk.event)

        response = ""
        for e in events[1:-1]:
            response += e.delta

        self.assertTrue("obama" in response.lower())

    async def test_tool_call_code_streaming(self):
        request = ChatCompletionRequest(
            model=self.valid_supported_model,
            messages=[
                UserMessage(
                    content="Write code to answer this question: What is the 100th prime number?",
                ),
            ],
            stream=True,
            tools=[ToolDefinition(tool_name=BuiltinTool.code_interpreter)],
        )
        iterator = self.api.chat_completion(request)
        events = []
        async for chunk in iterator:
            events.append(chunk.event)

        self.assertEqual(events[0].event_type, ChatCompletionResponseEventType.start)
        # last event is of type "complete"
        self.assertEqual(
            events[-1].event_type, ChatCompletionResponseEventType.complete
        )
        # last but one event should be eom with tool call
        self.assertEqual(
            events[-2].event_type, ChatCompletionResponseEventType.progress
        )
        self.assertEqual(events[-2].stop_reason, StopReason.end_of_turn)
        self.assertEqual(
            events[-2].delta.content.tool_name, BuiltinTool.code_interpreter
        )
