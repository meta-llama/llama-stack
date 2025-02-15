# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import logging
from typing import AsyncGenerator, Dict, List, Optional, Union

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel

from llama_stack.apis.common.content_types import (
    ImageContentItem,
    TextContentItem,
    TextDelta,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionResponse,
    CompletionResponseStreamChunk,
    Message,
    TokenLogProbs,
)
from llama_stack.models.llama.datatypes import (
    GreedySamplingStrategy,
    SamplingParams,
    StopReason,
    ToolCall,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    convert_image_content_to_url,
    decode_assistant_message,
)

logger = logging.getLogger(__name__)


class OpenAICompatCompletionChoiceDelta(BaseModel):
    content: str


class OpenAICompatLogprobs(BaseModel):
    text_offset: Optional[List[int]] = None

    token_logprobs: Optional[List[float]] = None

    tokens: Optional[List[str]] = None

    top_logprobs: Optional[List[Dict[str, float]]] = None


class OpenAICompatCompletionChoice(BaseModel):
    finish_reason: Optional[str] = None
    text: Optional[str] = None
    delta: Optional[OpenAICompatCompletionChoiceDelta] = None
    logprobs: Optional[OpenAICompatLogprobs] = None


class OpenAICompatCompletionResponse(BaseModel):
    choices: List[OpenAICompatCompletionChoice]


def get_sampling_strategy_options(params: SamplingParams) -> dict:
    options = {}
    if isinstance(params.strategy, GreedySamplingStrategy):
        options["temperature"] = 0.0
    elif isinstance(params.strategy, TopPSamplingStrategy):
        options["temperature"] = params.strategy.temperature
        options["top_p"] = params.strategy.top_p
    elif isinstance(params.strategy, TopKSamplingStrategy):
        options["top_k"] = params.strategy.top_k
    else:
        raise ValueError(f"Unsupported sampling strategy: {params.strategy}")

    return options


def get_sampling_options(params: SamplingParams) -> dict:
    options = {}
    if params:
        options.update(get_sampling_strategy_options(params))
        if params.max_tokens:
            options["max_tokens"] = params.max_tokens

        if params.repetition_penalty is not None and params.repetition_penalty != 1.0:
            options["repeat_penalty"] = params.repetition_penalty

    return options


def text_from_choice(choice) -> str:
    if hasattr(choice, "delta") and choice.delta:
        return choice.delta.content

    if hasattr(choice, "message"):
        return choice.message.content

    return choice.text


def get_stop_reason(finish_reason: str) -> StopReason:
    if finish_reason in ["stop", "eos"]:
        return StopReason.end_of_turn
    elif finish_reason == "eom":
        return StopReason.end_of_message
    elif finish_reason == "length":
        return StopReason.out_of_tokens

    return StopReason.out_of_tokens


def convert_openai_completion_logprobs(
    logprobs: Optional[OpenAICompatLogprobs],
) -> Optional[List[TokenLogProbs]]:
    if not logprobs:
        return None
    if hasattr(logprobs, "top_logprobs"):
        return [TokenLogProbs(logprobs_by_token=x) for x in logprobs.top_logprobs]

    # Together supports logprobs with top_k=1 only. This means for each token position,
    # they return only the logprobs for the selected token (vs. the top n most likely tokens).
    # Here we construct the response by matching the selected token with the logprobs.
    if logprobs.tokens and logprobs.token_logprobs:
        return [
            TokenLogProbs(logprobs_by_token={token: token_lp})
            for token, token_lp in zip(logprobs.tokens, logprobs.token_logprobs, strict=False)
        ]
    return None


def convert_openai_completion_logprobs_stream(text: str, logprobs: Optional[Union[float, OpenAICompatLogprobs]]):
    if logprobs is None:
        return None
    if isinstance(logprobs, float):
        # Adapt response from Together CompletionChoicesChunk
        return [TokenLogProbs(logprobs_by_token={text: logprobs})]
    if hasattr(logprobs, "top_logprobs"):
        return [TokenLogProbs(logprobs_by_token=x) for x in logprobs.top_logprobs]
    return None


def process_completion_response(response: OpenAICompatCompletionResponse) -> CompletionResponse:
    choice = response.choices[0]
    # drop suffix <eot_id> if present and return stop reason as end of turn
    if choice.text.endswith("<|eot_id|>"):
        return CompletionResponse(
            stop_reason=StopReason.end_of_turn,
            content=choice.text[: -len("<|eot_id|>")],
            logprobs=convert_openai_completion_logprobs(choice.logprobs),
        )
    # drop suffix <eom_id> if present and return stop reason as end of message
    if choice.text.endswith("<|eom_id|>"):
        return CompletionResponse(
            stop_reason=StopReason.end_of_message,
            content=choice.text[: -len("<|eom_id|>")],
            logprobs=convert_openai_completion_logprobs(choice.logprobs),
        )
    return CompletionResponse(
        stop_reason=get_stop_reason(choice.finish_reason),
        content=choice.text,
        logprobs=convert_openai_completion_logprobs(choice.logprobs),
    )


def process_chat_completion_response(
    response: OpenAICompatCompletionResponse,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    choice = response.choices[0]

    # TODO: This does not work well with tool calls for vLLM remote provider
    #   Ref: https://github.com/meta-llama/llama-stack/issues/1058
    raw_message = decode_assistant_message(text_from_choice(choice), get_stop_reason(choice.finish_reason))

    # NOTE: If we do not set tools in chat-completion request, we should not
    # expect the ToolCall in the response. Instead, we should return the raw
    # response from the model.
    if raw_message.tool_calls:
        if not request.tools:
            raw_message.tool_calls = []
            raw_message.content = text_from_choice(choice)
        else:
            # only return tool_calls if provided in the request
            new_tool_calls = []
            request_tools = {t.tool_name: t for t in request.tools}
            for t in raw_message.tool_calls:
                if t.tool_name in request_tools:
                    new_tool_calls.append(t)
                else:
                    logger.warning(f"Tool {t.tool_name} not found in request tools")

            if len(new_tool_calls) < len(raw_message.tool_calls):
                raw_message.tool_calls = new_tool_calls
                raw_message.content = text_from_choice(choice)

    return ChatCompletionResponse(
        completion_message=CompletionMessage(
            content=raw_message.content,
            stop_reason=raw_message.stop_reason,
            tool_calls=raw_message.tool_calls,
        ),
        logprobs=None,
    )


async def process_completion_stream_response(
    stream: AsyncGenerator[OpenAICompatCompletionResponse, None],
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
            logprobs=convert_openai_completion_logprobs_stream(text, choice.logprobs),
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
    stream: AsyncGenerator[OpenAICompatCompletionResponse, None],
    request: ChatCompletionRequest,
) -> AsyncGenerator:
    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.start,
            delta=TextDelta(text=""),
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
        if not text:
            # Sometimes you get empty chunks from providers
            continue

        # check if its a tool call ( aka starts with <|python_tag|> )
        if not ipython and text.startswith("<|python_tag|>"):
            ipython = True
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=ToolCallDelta(
                        tool_call="",
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
                tool_call=text,
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
                    delta=TextDelta(text=text),
                    stop_reason=stop_reason,
                )
            )

    # parse tool calls and report errors
    message = decode_assistant_message(buffer, stop_reason)

    parsed_tool_calls = len(message.tool_calls) > 0
    if ipython and not parsed_tool_calls:
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.progress,
                delta=ToolCallDelta(
                    tool_call="",
                    parse_status=ToolCallParseStatus.failed,
                ),
                stop_reason=stop_reason,
            )
        )

    request_tools = {t.tool_name: t for t in request.tools}
    for tool_call in message.tool_calls:
        if tool_call.tool_name in request_tools:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=ToolCallDelta(
                        tool_call=tool_call,
                        parse_status=ToolCallParseStatus.succeeded,
                    ),
                    stop_reason=stop_reason,
                )
            )
        else:
            logger.warning(f"Tool {tool_call.tool_name} not found in request tools")
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=ToolCallDelta(
                        # Parsing tool call failed due to tool call not being found in request tools,
                        # We still add the raw message text inside tool_call for responding back to the user
                        tool_call=buffer,
                        parse_status=ToolCallParseStatus.failed,
                    ),
                    stop_reason=stop_reason,
                )
            )

    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.complete,
            delta=TextDelta(text=""),
            stop_reason=stop_reason,
        )
    )


async def convert_message_to_openai_dict(message: Message, download: bool = False) -> dict:
    async def _convert_content(content) -> dict:
        if isinstance(content, ImageContentItem):
            return {
                "type": "image_url",
                "image_url": {
                    "url": await convert_image_content_to_url(content, download=download),
                },
            }
        else:
            text = content.text if isinstance(content, TextContentItem) else content
            assert isinstance(text, str)
            return {"type": "text", "text": text}

    if isinstance(message.content, list):
        content = [await _convert_content(c) for c in message.content]
    else:
        content = [await _convert_content(message.content)]

    return {
        "role": message.role,
        "content": content,
    }


class UnparseableToolCall(BaseModel):
    """
    A ToolCall with arguments that are not valid JSON.
    Mirrors the ToolCall schema, but with arguments as a string.
    """

    call_id: str = ""
    tool_name: str = ""
    arguments: str = ""


def convert_tool_call(
    tool_call: ChatCompletionMessageToolCall,
) -> Union[ToolCall, UnparseableToolCall]:
    """
    Convert a ChatCompletionMessageToolCall tool call to either a
    ToolCall or UnparseableToolCall. Returns an UnparseableToolCall
    if the tool call is not valid ToolCall.
    """
    try:
        valid_tool_call = ToolCall(
            call_id=tool_call.id,
            tool_name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
    except Exception as e:
        return UnparseableToolCall(
            call_id=tool_call.id or "",
            tool_name=tool_call.function.name or "",
            arguments=tool_call.function.arguments or "",
        )

    return valid_tool_call
