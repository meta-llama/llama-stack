# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import unittest
from unittest import mock

from cerebras.cloud.sdk.types.chat.completion_create_response import (
    ChatChunkResponse,
    ChatCompletion,
)
from llama_models.llama3.api.datatypes import (
    BuiltinTool,
    CompletionMessage,
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
from llama_stack.providers.adapters.inference.cerebras import get_adapter_impl
from llama_stack.providers.adapters.inference.cerebras.config import CerebrasImplConfig


class CerebrasInferenceTests(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        cerebras_config = CerebrasImplConfig(api_key="foobar")

        # setup Cerebras
        self.api = await get_adapter_impl(cerebras_config, {})
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
        self.valid_supported_model = "Llama3.1-70B-Instruct"

    async def asyncTearDown(self):
        await self.api.shutdown()

    async def test_text(self):
        with mock.patch.object(
            self.api.client.chat.completions, "create"
        ) as mock_completion:
            mock_completion.return_value = ChatCompletion(
                **{
                    "id": "chatcmpl-1f0a31de-a615-4ef5-a355-c72b840710d0",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {
                                "content": "The capital of France is Paris.",
                                "role": "assistant",
                            },
                        }
                    ],
                    "created": 1729026294,
                    "model": "llama3.1-70b",
                    "system_fingerprint": "fp_97b75e13af",
                    "object": "chat.completion",
                    "usage": {
                        "prompt_tokens": 17,
                        "completion_tokens": 8,
                        "total_tokens": 25,
                    },
                    "time_info": {
                        "queue_time": 2.702e-05,
                        "prompt_time": 0.0013021605714285715,
                        "completion_time": 0.0039899714285714285,
                        "total_time": 0.01815319061279297,
                        "created": 1729026294,
                    },
                }
            )

            request = ChatCompletionRequest(
                model=self.valid_supported_model,
                messages=[
                    UserMessage(
                        content="What is the capital of France?",
                    ),
                ],
                stream=False,
            )
            response = self.api.chat_completion(
                request.model,
                request.messages,
                request.sampling_params,
                request.tools,
                request.tool_choice,
                request.tool_prompt_format,
                request.stream,
                request.logprobs,
            )

            result = response.completion_message.content
            self.assertTrue("Paris" in result, result)

    async def test_text_streaming(self):
        events = [
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {"role": "assistant"}, "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {"content": "The"}, "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {"content": " capital"}, "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {"content": " of"}, "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {"content": " France"}, "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {"content": " is"}, "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {"content": " Paris"}, "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {"content": "."}, "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-f908bda3-eaa6-4148-ada8-689631b1e7c7",
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "created": 1729094696,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 8,
                    "total_tokens": 25,
                },
                "time_info": {
                    "queue_time": 2.568e-05,
                    "prompt_time": 0.001300800857142857,
                    "completion_time": 0.0039863531428571426,
                    "total_time": 0.020612478256225586,
                    "created": 1729094696,
                },
            },
        ]

        with mock.patch.object(
            self.api.client.chat.completions, "create"
        ) as mock_completion_stream:
            mock_completion_stream.return_value = [
                ChatChunkResponse(**event) for event in events
            ]

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
                # print(f"{chunk.event.event_type:<40} | {str(chunk.event.stop_reason):<26} | {chunk.event.delta} ")

            self.assertEqual(
                events[0].event_type, ChatCompletionResponseEventType.start
            )
            self.assertEqual(
                events[-1].event_type, ChatCompletionResponseEventType.complete
            )

            response = ""
            for e in events[1:-1]:
                response += e.delta

            self.assertTrue("Paris" in response, response)

    async def test_custom_tool_call(self):
        with mock.patch.object(
            self.api.client.chat.completions, "create"
        ) as mock_completion:
            mock_completion.return_value = ChatCompletion(
                **{
                    "id": "chatcmpl-f673ee3b-2598-4cd4-8952-35f183564441",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "index": 0,
                            "message": {
                                "tool_calls": [
                                    {
                                        "id": "253ac93a8",
                                        "type": "function",
                                        "function": {
                                            "name": "get_boiling_point",
                                            "arguments": '{"liquid_name": "polyjuice", "celcius": "True"}',
                                        },
                                    }
                                ],
                                "role": "assistant",
                            },
                        }
                    ],
                    "created": 1729095512,
                    "model": "llama3.1-70b",
                    "system_fingerprint": "fp_97b75e13af",
                    "object": "chat.completion",
                    "usage": {
                        "prompt_tokens": 193,
                        "completion_tokens": 14,
                        "total_tokens": 207,
                    },
                    "time_info": {
                        "queue_time": 2.651e-05,
                        "prompt_time": 0.008480784000000002,
                        "completion_time": 0.024744930000000002,
                        "total_time": 0.03530120849609375,
                        "created": 1729095512,
                    },
                }
            )

            request = ChatCompletionRequest(
                tool_choice=ToolChoice.required,
                model=self.valid_supported_model,
                messages=[
                    UserMessage(
                        content="Use provided function to find the boiling point of polyjuice?",
                    ),
                ],
                stream=False,
                tools=[self.custom_tool_defn],
            )
            response = self.api.chat_completion(
                request.model,
                request.messages,
                request.sampling_params,
                request.tools,
                request.tool_choice,
                request.tool_prompt_format,
                request.stream,
                request.logprobs,
            )

            completion_message = response.completion_message

            self.assertEqual(completion_message.content, "")

            self.assertEqual(completion_message.stop_reason, StopReason.end_of_turn)

            self.assertEqual(
                len(completion_message.tool_calls), 1, completion_message.tool_calls
            )
            self.assertEqual(
                completion_message.tool_calls[0].tool_name, "get_boiling_point"
            )

            args = completion_message.tool_calls[0].arguments
            self.assertTrue(isinstance(args, dict))
            self.assertTrue(args["liquid_name"], "polyjuice")

    async def test_tool_call_streaming(self):
        events = [
            {
                "id": "chatcmpl-1e573d82-bd76-496b-aa01-18faed024a1d",
                "choices": [{"delta": {"role": "assistant"}, "index": 0}],
                "created": 1729101621,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-1e573d82-bd76-496b-aa01-18faed024a1d",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "df0a5e087",
                                    "type": "function",
                                    "function": {
                                        "name": "brave_search",
                                        "arguments": '{"query": "current US President"}',
                                    },
                                }
                            ]
                        },
                        "index": 0,
                    }
                ],
                "created": 1729101621,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
            },
            {
                "id": "chatcmpl-1e573d82-bd76-496b-aa01-18faed024a1d",
                "choices": [{"delta": {}, "finish_reason": "tool_calls", "index": 0}],
                "created": 1729101621,
                "model": "llama3.1-70b",
                "system_fingerprint": "fp_97b75e13af",
                "object": "chat.completion.chunk",
                "usage": {
                    "prompt_tokens": 193,
                    "completion_tokens": 14,
                    "total_tokens": 207,
                },
                "time_info": {
                    "queue_time": 0.00001926,
                    "prompt_time": 0.008447145307692307,
                    "completion_time": 0.024725255692307692,
                    "total_time": 0.04976296424865723,
                    "created": 1729101621,
                },
            },
        ]
        with mock.patch.object(
            self.api.client.chat.completions, "create"
        ) as mock_completion_stream:
            mock_completion_stream.return_value = [
                ChatChunkResponse(**event) for event in events
            ]

            request = ChatCompletionRequest(
                model=self.valid_supported_model,
                tool_choice=ToolChoice.required,
                messages=[
                    UserMessage(
                        content="Who is the current US President?",
                    ),
                ],
                stream=True,
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

            events = []
            async for chunk in iterator:
                # print(f"{chunk.event.event_type:<40} | {str(chunk.event.stop_reason):<26} | {chunk.event.delta} ")
                events.append(chunk.event)

            self.assertEqual(
                events[0].event_type, ChatCompletionResponseEventType.start
            )
            # last event is of type "complete"
            self.assertEqual(
                events[-1].event_type, ChatCompletionResponseEventType.complete
            )
            # last but one event should be eom with tool call
            self.assertEqual(
                events[-2].event_type, ChatCompletionResponseEventType.progress
            )
            self.assertEqual(events[-1].stop_reason, StopReason.end_of_turn)
            self.assertEqual(
                events[-2].delta.content.tool_name, BuiltinTool.brave_search
            )

    async def test_multi_turn_non_streaming(self):
        with mock.patch.object(
            self.api.client.chat.completions, "create"
        ) as mock_completion:
            mock_completion.return_value = ChatCompletion(
                **{
                    "id": "chatcmpl-1f0a31de-a615-4ef5-a355-c72b840710d0",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "index": 0,
                            "message": {
                                "content": "The 44th president of the United States was Barack Obama.",
                                "role": "assistant",
                            },
                        }
                    ],
                    "created": 1729026294,
                    "model": "llama3.1-70b",
                    "system_fingerprint": "fp_97b75e13af",
                    "object": "chat.completion",
                    "usage": {
                        "prompt_tokens": 17,
                        "completion_tokens": 8,
                        "total_tokens": 25,
                    },
                    "time_info": {
                        "queue_time": 2.702e-05,
                        "prompt_time": 0.0013021605714285715,
                        "completion_time": 0.0039899714285714285,
                        "total_time": 0.01815319061279297,
                        "created": 1729026294,
                    },
                }
            )

            request = ChatCompletionRequest(
                model=self.valid_supported_model,
                messages=[
                    UserMessage(
                        content="Search the web and tell me who the "
                        "44th president of the United States was",
                    ),
                    CompletionMessage(
                        content="",
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
                        content="Barack Obama",
                    ),
                ],
                stream=False,
                tools=[ToolDefinition(tool_name=BuiltinTool.brave_search)],
            )
            response = self.api.chat_completion(
                request.model,
                request.messages,
                request.sampling_params,
                request.tools,
                request.tool_choice,
                request.tool_prompt_format,
                request.stream,
                request.logprobs,
            )

            completion_message = response.completion_message

            self.assertTrue(
                completion_message.stop_reason
                in {
                    StopReason.end_of_turn,
                    StopReason.end_of_message,
                }
            )

            self.assertTrue("obama" in completion_message.content.lower())
