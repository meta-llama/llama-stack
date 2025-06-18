# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from llama_stack.apis.inference import (
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIChoiceLogprobs,
    OpenAIMessageParam,
)
from llama_stack.providers.utils.inference.inference_store import InferenceStore


async def stream_and_store_openai_completion(
    provider_stream: AsyncIterator[OpenAIChatCompletionChunk],
    model: str,
    store: InferenceStore,
    input_messages: list[OpenAIMessageParam],
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """
    Wraps a provider's stream, yields chunks, and stores the full completion at the end.
    """
    id = None
    created = None
    choices_data: dict[int, dict[str, Any]] = {}

    try:
        async for chunk in provider_stream:
            if id is None and chunk.id:
                id = chunk.id
            if created is None and chunk.created:
                created = chunk.created

            if chunk.choices:
                for choice_delta in chunk.choices:
                    idx = choice_delta.index
                    if idx not in choices_data:
                        choices_data[idx] = {
                            "content_parts": [],
                            "tool_calls_builder": {},
                            "finish_reason": None,
                            "logprobs_content_parts": [],
                        }
                    current_choice_data = choices_data[idx]

                    if choice_delta.delta:
                        delta = choice_delta.delta
                        if delta.content:
                            current_choice_data["content_parts"].append(delta.content)
                        if delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                tc_idx = tool_call_delta.index
                                if tc_idx not in current_choice_data["tool_calls_builder"]:
                                    # Initialize with correct structure for _ToolCallBuilderData
                                    current_choice_data["tool_calls_builder"][tc_idx] = {
                                        "id": None,
                                        "type": "function",
                                        "function_name_parts": [],
                                        "function_arguments_parts": [],
                                    }
                                builder = current_choice_data["tool_calls_builder"][tc_idx]
                                if tool_call_delta.id:
                                    builder["id"] = tool_call_delta.id
                                if tool_call_delta.type:
                                    builder["type"] = tool_call_delta.type
                                if tool_call_delta.function:
                                    if tool_call_delta.function.name:
                                        builder["function_name_parts"].append(tool_call_delta.function.name)
                                    if tool_call_delta.function.arguments:
                                        builder["function_arguments_parts"].append(tool_call_delta.function.arguments)
                    if choice_delta.finish_reason:
                        current_choice_data["finish_reason"] = choice_delta.finish_reason
                    if choice_delta.logprobs and choice_delta.logprobs.content:
                        # Ensure that we are extending with the correct type
                        current_choice_data["logprobs_content_parts"].extend(choice_delta.logprobs.content)
            yield chunk
    finally:
        if id:
            assembled_choices: list[OpenAIChoice] = []
            for choice_idx, choice_data in choices_data.items():
                content_str = "".join(choice_data["content_parts"])
                assembled_tool_calls: list[OpenAIChatCompletionToolCall] = []
                if choice_data["tool_calls_builder"]:
                    for tc_build_data in choice_data["tool_calls_builder"].values():
                        if tc_build_data["id"]:
                            func_name = "".join(tc_build_data["function_name_parts"])
                            func_args = "".join(tc_build_data["function_arguments_parts"])
                            assembled_tool_calls.append(
                                OpenAIChatCompletionToolCall(
                                    id=tc_build_data["id"],
                                    type=tc_build_data["type"],  # No or "function" needed, already set
                                    function=OpenAIChatCompletionToolCallFunction(name=func_name, arguments=func_args),
                                )
                            )
                message = OpenAIAssistantMessageParam(
                    role="assistant",
                    content=content_str if content_str else None,
                    tool_calls=assembled_tool_calls if assembled_tool_calls else None,
                )
                logprobs_content = choice_data["logprobs_content_parts"]
                final_logprobs = OpenAIChoiceLogprobs(content=logprobs_content) if logprobs_content else None

                assembled_choices.append(
                    OpenAIChoice(
                        finish_reason=choice_data["finish_reason"],
                        index=choice_idx,
                        message=message,
                        logprobs=final_logprobs,
                    )
                )

            final_response = OpenAIChatCompletion(
                id=id,
                choices=assembled_choices,
                created=created or int(datetime.now(UTC).timestamp()),
                model=model,
                object="chat.completion",
            )
            await store.store_chat_completion(final_response, input_messages)
