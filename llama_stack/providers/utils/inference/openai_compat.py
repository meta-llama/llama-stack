# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

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


def get_sampling_options(params: SamplingParams) -> dict:
    options = {}
    if params:
        for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
            if getattr(params, attr):
                options[attr] = getattr(params, attr)

        if params.repetition_penalty is not None and params.repetition_penalty != 1.0:
            options["repeat_penalty"] = params.repetition_penalty

    return options


def text_from_choice(choice) -> str:
    if hasattr(choice, "delta") and choice.delta:
        return choice.delta.content

    return choice.text


def get_stop_reason(finish_reason: str) -> StopReason:
    if finish_reason in ["stop", "eos"]:
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
    # drop suffix <eot_id> if present and return stop reason as end of turn
    if choice.text.endswith("<|eot_id|>"):
        return CompletionResponse(
            stop_reason=StopReason.end_of_turn,
            content=choice.text[: -len("<|eot_id|>")],
        )
    # drop suffix <eom_id> if present and return stop reason as end of message
    if choice.text.endswith("<|eom_id|>"):
        return CompletionResponse(
            stop_reason=StopReason.end_of_message,
            content=choice.text[: -len("<|eom_id|>")],
        )
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
        if finish_reason:
            if finish_reason in ["stop", "eos", "eos_token"]:
                stop_reason = StopReason.end_of_turn
            elif finish_reason == "length":
                stop_reason = StopReason.out_of_tokens
            break

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
            if stop_reason is None and finish_reason in ["stop", "eos", "eos_token"]:
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

        if ipython:
            buffer += text
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
            buffer += text
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
