# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from typing import AsyncGenerator, Optional

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import StopReason

from llama_stack.apis.inference import *  # noqa: F403

from pydantic import BaseModel


class OpenAICompatCompletionChoiceDelta(BaseModel):
    content: str


class OpenAICompatCompletionChoice(BaseModel):
    finish_reason: Optional[str] = None
    text: Optional[str] = None
    delta: Optional[OpenAICompatCompletionChoiceDelta] = None


class OpenAICompatCompletionResponse(BaseModel):
    choices: List[OpenAICompatCompletionChoice]


def get_sampling_options(request: ChatCompletionRequest) -> dict:
    options = {}
    if params := request.sampling_params:
        for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
            if getattr(params, attr):
                options[attr] = getattr(params, attr)

        if params.repetition_penalty is not None and params.repetition_penalty != 1.0:
            options["repeat_penalty"] = params.repetition_penalty

    return options


def text_from_choice(choice) -> str:
    if hasattr(choice, "delta") and choice.delta:
        return choice.delta.content or ""

    if hasattr(choice, "message") and choice.message:
        return choice.message.content or ""

    if hasattr(choice, "text"):
        return choice.text or ""

    return ""


def tool_calls_from_choice(choice) -> List[ToolCall]:
    tool_calls = []

    if choice.message and choice.message.tool_calls:
        for tool_call in choice.message.tool_calls:
            tool_calls.append(
                ToolCall(
                    call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments),
                )
            )

    return tool_calls


def tool_call_deltas_from_choice(choice) -> List[ToolCallDelta]:
    tool_call_deltas = []

    if choice.delta and choice.delta.tool_calls:
        for tool_call in choice.delta.tool_calls:
            tool_call_deltas.append(
                ToolCallDelta(
                    content=ToolCall(
                        call_id=tool_call.id,
                        tool_name=tool_call.function.name,
                        arguments=json.loads(tool_call.function.arguments),
                    ),
                    parse_status=ToolCallParseStatus.in_progress,
                )
            )

    return tool_call_deltas


def get_stop_reason(finish_reason: str) -> StopReason:  
    if finish_reason in ["stop", "eos", "tool_calls"]:
        return StopReason.end_of_turn
    elif finish_reason == "eom":
        return StopReason.end_of_message
    elif finish_reason == "length":
        return StopReason.out_of_tokens

    return StopReason.out_of_tokens


def process_completion_response(
    response: OpenAICompatCompletionResponse, formatter: ChatFormat
) -> CompletionResponse:
    choice = response.choices[0]

    return CompletionResponse(
        stop_reason=get_stop_reason(choice.finish_reason),
        content=choice.text,
    )


def process_chat_completion_response(
    response: OpenAICompatCompletionResponse, formatter: ChatFormat
) -> ChatCompletionResponse:
    choice = response.choices[0]

    completion_message = formatter.decode_assistant_message_from_content(
        text_from_choice(choice), get_stop_reason(choice.finish_reason)
    )

    # According to the OpenAI spec, tool calls are embedded as a field in the response object.
    completion_message.tool_calls += tool_calls_from_choice(choice)

    return ChatCompletionResponse(
        completion_message=completion_message,
        logprobs=None,
    )


async def process_completion_stream_response(
    stream: AsyncGenerator[OpenAICompatCompletionResponse, None], formatter: ChatFormat
) -> AsyncGenerator:

    stop_reason = None

    async for chunk in stream:
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason:
            if finish_reason in ["stop", "eos", "eos_token"]:
                stop_reason = StopReason.end_of_turn
            elif finish_reason == "length":
                stop_reason = StopReason.out_of_tokens
            break

        text = text_from_choice(choice)
        if text == "<|eot_id|>":
            stop_reason = StopReason.end_of_turn
            text = ""
            continue
        elif text == "<|eom_id|>":
            stop_reason = StopReason.end_of_message
            text = ""
            continue
        yield CompletionResponseStreamChunk(
            delta=text,
            stop_reason=stop_reason,
        )

    yield CompletionResponseStreamChunk(
        delta="",
        stop_reason=stop_reason,
    )


async def process_chat_completion_stream_response(
    stream: AsyncGenerator[OpenAICompatCompletionResponse, None], formatter: ChatFormat
) -> AsyncGenerator:
    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.start,
            delta="",
        )
    )

    buffer = ""
    ipython = False
    stop_reason = None

    async for chunk in stream:
        choice = chunk.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason:
            if stop_reason is None and finish_reason in [
                "stop",
                "eos",
                "eos_token",
                "tool_calls",
            ]:
                stop_reason = StopReason.end_of_turn
            elif stop_reason is None and finish_reason == "length":
                stop_reason = StopReason.out_of_tokens
            break

        text = text_from_choice(choice)
        # check if its a tool call ( aka starts with <|python_tag|> )
        if not ipython and text.startswith("<|python_tag|>"):
            ipython = True
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=ToolCallDelta(
                        content="",
                        parse_status=ToolCallParseStatus.started,
                    ),
                )
            )
            buffer += text
            continue

        if text == "<|eot_id|>":
            stop_reason = StopReason.end_of_turn
            text = ""
            continue
        elif text == "<|eom_id|>":
            stop_reason = StopReason.end_of_message
            text = ""
            continue

        buffer += text
        if tool_call_deltas := tool_call_deltas_from_choice(choice):
            for delta in tool_call_deltas:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=delta,
                        stop_reason=stop_reason,
                    )
                )

        elif ipython:
            delta = ToolCallDelta(
                content=text,
                parse_status=ToolCallParseStatus.in_progress,
            )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=delta,
                    stop_reason=stop_reason,
                )
            )
        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=text,
                    stop_reason=stop_reason,
                )
            )

    # parse tool calls and report errors
    message = formatter.decode_assistant_message_from_content(buffer, stop_reason)

    parsed_tool_calls = len(message.tool_calls) > 0
    if ipython and not parsed_tool_calls:
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.progress,
                delta=ToolCallDelta(
                    content="",
                    parse_status=ToolCallParseStatus.failure,
                ),
                stop_reason=stop_reason,
            )
        )

    for tool_call in message.tool_calls:
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.progress,
                delta=ToolCallDelta(
                    content=tool_call,
                    parse_status=ToolCallParseStatus.success,
                ),
                stop_reason=stop_reason,
            )
        )

    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.complete,
            delta="",
            stop_reason=stop_reason,
        )
    )
