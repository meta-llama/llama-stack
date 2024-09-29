# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unittest
from unittest import mock

from llama_models.llama3.api.datatypes import (
    BuiltinTool,
    CompletionMessage,
    SamplingParams,
    SamplingStrategy,
    StopReason,
    ToolCall,
    ToolChoice,
    ToolDefinition,
    ToolParamDefinition,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.inference.inference import (
    ChatCompletionRequest,
    ChatCompletionResponseEventType,
)
from llama_stack.providers.adapters.inference.bedrock import get_adapter_impl
from llama_stack.providers.adapters.inference.bedrock.config import BedrockConfig


class BedrockInferenceTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        bedrock_config = BedrockConfig()

        # setup Bedrock
        self.api = await get_adapter_impl(bedrock_config, {})
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
        self.valid_supported_model = "Meta-Llama3.1-8B-Instruct"

    async def asyncTearDown(self):
        await self.api.shutdown()

    async def test_text(self):
        with mock.patch.object(self.api.client, "converse") as mock_converse:
            mock_converse.return_value = {
                "ResponseMetadata": {
                    "RequestId": "8ad04352-cd81-4946-b811-b434e546385d",
                    "HTTPStatusCode": 200,
                    "HTTPHeaders": {},
                    "RetryAttempts": 0,
                },
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "\n\nThe capital of France is Paris."}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 21, "outputTokens": 9, "totalTokens": 30},
                "metrics": {"latencyMs": 307},
            }
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
                request.model,
                request.messages,
                request.sampling_params,
                request.tools,
                request.tool_choice,
                request.tool_prompt_format,
                request.stream,
                request.logprobs,
            )
            async for r in iterator:
                response = r
            print(response.completion_message.content)
            self.assertTrue("Paris" in response.completion_message.content[0])
            self.assertEqual(
                response.completion_message.stop_reason, StopReason.end_of_turn
            )

    async def test_tool_call(self):
        with mock.patch.object(self.api.client, "converse") as mock_converse:
            mock_converse.return_value = {
                "ResponseMetadata": {
                    "RequestId": "ec9da6a4-656b-4343-9e1f-71dac79cbf53",
                    "HTTPStatusCode": 200,
                    "HTTPHeaders": {},
                    "RetryAttempts": 0,
                },
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "name": "brave_search",
                                    "toolUseId": "tooluse_d49kUQ3rTc6K_LPM-w96MQ",
                                    "input": {"query": "current US President"},
                                }
                            }
                        ],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 48, "outputTokens": 81, "totalTokens": 129},
                "metrics": {"latencyMs": 1236},
            }
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
            iterator = self.api.chat_completion(
                request.model,
                request.messages,
                request.sampling_params,
                request.tools,
                request.tool_choice,
                request.tool_prompt_format,
                request.stream,
                request.logprobs,
            )
            async for r in iterator:
                response = r

            completion_message = response.completion_message

            self.assertEqual(len(completion_message.content), 0)
            self.assertEqual(completion_message.stop_reason, StopReason.end_of_turn)

            self.assertEqual(
                len(completion_message.tool_calls), 1, completion_message.tool_calls
            )
            self.assertEqual(
                completion_message.tool_calls[0].tool_name, BuiltinTool.brave_search
            )
            self.assertTrue(
                "president"
                in completion_message.tool_calls[0].arguments["query"].lower()
            )

    async def test_custom_tool(self):
        with mock.patch.object(self.api.client, "converse") as mock_converse:
            mock_converse.return_value = {
                "ResponseMetadata": {
                    "RequestId": "243c4316-0965-4b79-a145-2d9ac6b4e9ad",
                    "HTTPStatusCode": 200,
                    "HTTPHeaders": {},
                    "RetryAttempts": 0,
                },
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tooluse_7DViuqxXS6exL8Yug9Apjw",
                                    "name": "get_boiling_point",
                                    "input": {
                                        "liquid_name": "polyjuice",
                                        "celcius": "True",
                                    },
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 110, "outputTokens": 37, "totalTokens": 147},
                "metrics": {"latencyMs": 743},
            }

            request = ChatCompletionRequest(
                model=self.valid_supported_model,
                messages=[
                    UserMessage(
                        content="Use provided function to find the boiling point of polyjuice?",
                    ),
                ],
                stream=False,
                tools=[self.custom_tool_defn],
                tool_choice=ToolChoice.required,
            )
            iterator = self.api.chat_completion(
                request.model,
                request.messages,
                request.sampling_params,
                request.tools,
                request.tool_choice,
                request.tool_prompt_format,
                request.stream,
                request.logprobs,
            )
            async for r in iterator:
                response = r

            completion_message = response.completion_message

            self.assertEqual(len(completion_message.content), 0)
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
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"delta": {"text": "\n\n"}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": "The"}, "contentBlockIndex": 0}},
            {
                "contentBlockDelta": {
                    "delta": {"text": " capital"},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockDelta": {"delta": {"text": " of"}, "contentBlockIndex": 0}},
            {
                "contentBlockDelta": {
                    "delta": {"text": " France"},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockDelta": {"delta": {"text": " is"}, "contentBlockIndex": 0}},
            {
                "contentBlockDelta": {
                    "delta": {"text": " Paris"},
                    "contentBlockIndex": 0,
                }
            },
            {"contentBlockDelta": {"delta": {"text": "."}, "contentBlockIndex": 0}},
            {"contentBlockDelta": {"delta": {"text": ""}, "contentBlockIndex": 0}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"messageStop": {"stopReason": "end_turn"}},
            {
                "metadata": {
                    "usage": {"inputTokens": 21, "outputTokens": 9, "totalTokens": 30},
                    "metrics": {"latencyMs": 1},
                }
            },
        ]

        with mock.patch.object(
            self.api.client, "converse_stream"
        ) as mock_converse_stream:
            mock_converse_stream.return_value = {"stream": events}
            request = ChatCompletionRequest(
                model=self.valid_supported_model,
                messages=[
                    UserMessage(
                        content="What is the capital of France?",
                    ),
                ],
                stream=True,
            )
            iterator = self.api.chat_completion(
                request.model,
                request.messages,
                request.sampling_params,
                request.tools,
                request.tool_choice,
                request.tool_prompt_format,
                request.stream,
                request.logprobs,
            )
            events = []
            async for chunk in iterator:
                events.append(chunk.event)

            response = ""
            for e in events[1:-1]:
                response += e.delta

            self.assertEqual(
                events[0].event_type, ChatCompletionResponseEventType.start
            )
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

    def test_resolve_bedrock_model(self):
        bedrock_model = self.api.resolve_bedrock_model(self.valid_supported_model)
        self.assertEqual(bedrock_model, "meta.llama3-1-8b-instruct-v1:0")

        invalid_model = "Meta-Llama3.1-8B"
        with self.assertRaisesRegex(
            AssertionError, f"Unsupported model: {invalid_model}"
        ):
            self.api.resolve_bedrock_model(invalid_model)

    async def test_bedrock_chat_inference_config(self):
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
        options = self.api.get_bedrock_inference_config(request.sampling_params)
        self.assertEqual(
            options,
            {
                "temperature": 1.0,
                "topP": 0.99,
            },
        )

    async def test_multi_turn_non_streaming(self):
        with mock.patch.object(self.api.client, "converse") as mock_converse:
            mock_converse.return_value = {
                "ResponseMetadata": {
                    "RequestId": "4171abf1-a5f4-4eee-bb12-0e472a73bdbe",
                    "HTTPStatusCode": 200,
                    "HTTPHeaders": {},
                    "RetryAttempts": 0,
                },
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "text": "\nThe 44th president of the United States was Barack Obama."
                            }
                        ],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 723, "outputTokens": 15, "totalTokens": 738},
                "metrics": {"latencyMs": 449},
            }

            request = ChatCompletionRequest(
                model=self.valid_supported_model,
                messages=[
                    UserMessage(
                        content="Search the web and tell me who the "
                        "44th president of the United States was",
                    ),
                    CompletionMessage(
                        content=[],
                        stop_reason=StopReason.end_of_turn,
                        tool_calls=[
                            ToolCall(
                                call_id="1",
                                tool_name=BuiltinTool.brave_search,
                                arguments={
                                    "query": "44th president of the United States"
                                },
                            )
                        ],
                    ),
                    ToolResponseMessage(
                        call_id="1",
                        tool_name=BuiltinTool.brave_search,
                        content='{"query": "44th president of the United States", "top_k": [{"title": "Barack Obama | The White House", "url": "https://www.whitehouse.gov/about-the-white-house/presidents/barack-obama/", "description": "<strong>Barack Obama</strong> served as the 44th President of the United States. His story is the American story \\u2014 values from the heartland, a middle-class upbringing in a strong family, hard work and education as the means of getting ahead, and the conviction that a life so blessed should be lived in service ...", "type": "search_result"}, {"title": "Barack Obama \\u2013 The White House", "url": "https://trumpwhitehouse.archives.gov/about-the-white-house/presidents/barack-obama/", "description": "After working his way through college with the help of scholarships and student loans, <strong>President Obama</strong> moved to Chicago, where he worked with a group of churches to help rebuild communities devastated by the closure of local steel plants.", "type": "search_result"}, [{"type": "video_result", "url": "https://www.instagram.com/reel/CzMZbJmObn9/", "title": "Fifteen years ago, on Nov. 4, Barack Obama was elected as ...", "description": ""}, {"type": "video_result", "url": "https://video.alexanderstreet.com/watch/the-44th-president-barack-obama?context=channel:barack-obama", "title": "The 44th President (Barack Obama) - Alexander Street, a ...", "description": "You need to enable JavaScript to run this app"}, {"type": "video_result", "url": "https://www.youtube.com/watch?v=iyL7_2-em5k", "title": "Barack Obama for Kids | Learn about the life and contributions ...", "description": "Enjoy the videos and music you love, upload original content, and share it all with friends, family, and the world on YouTube."}, {"type": "video_result", "url": "https://www.britannica.com/video/172743/overview-Barack-Obama", "title": "President of the United States of America Barack Obama | Britannica", "description": "[NARRATOR] Barack Obama was elected the 44th president of the United States in 2008, becoming the first African American to hold the office. Obama vowed to bring change to the political system."}, {"type": "video_result", "url": "https://www.youtube.com/watch?v=rvr2g8-5dcE", "title": "The 44th President: In His Own Words - Toughest Day | Special ...", "description": "President Obama reflects on his toughest day in the Presidency and seeing Secret Service cry for the first time. Watch the premiere of The 44th President: In..."}]]}',
                    ),
                ],
                stream=False,
                tools=[ToolDefinition(tool_name=BuiltinTool.brave_search)],
            )
            iterator = self.api.chat_completion(
                request.model,
                request.messages,
                request.sampling_params,
                request.tools,
                request.tool_choice,
                request.tool_prompt_format,
                request.stream,
                request.logprobs,
            )
            async for r in iterator:
                response = r

            completion_message = response.completion_message

            self.assertEqual(len(completion_message.content), 1)
            self.assertTrue(
                completion_message.stop_reason
                in {
                    StopReason.end_of_turn,
                    StopReason.end_of_message,
                }
            )

            self.assertTrue("obama" in completion_message.content[0].lower())
