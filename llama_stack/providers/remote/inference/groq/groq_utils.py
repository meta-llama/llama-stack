# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import warnings
from typing import AsyncGenerator, Literal, Union

from groq import Stream
from groq.types.chat.chat_completion import ChatCompletion
from groq.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from groq.types.chat.chat_completion_chunk import ChatCompletionChunk
from groq.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from groq.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
from groq.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from groq.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from groq.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from groq.types.chat.completion_create_params import CompletionCreateParams
from groq.types.shared.function_definition import FunctionDefinition

from llama_models.llama3.api.datatypes import ToolParamDefinition

from pydantic import BaseModel

from llama_stack.apis.common.content_types import (
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
    Message,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_strategy_options,
)


def convert_chat_completion_request(
    request: ChatCompletionRequest,
) -> CompletionCreateParams:
    """
    Convert a ChatCompletionRequest to a Groq API-compatible dictionary.
    Warns client if request contains unsupported features.
    """

    if request.logprobs:
        # Groq doesn't support logprobs at the time of writing
        warnings.warn("logprobs are not supported yet")

    if request.response_format:
        # Groq's JSON mode is beta at the time of writing
        warnings.warn("response_format is not supported yet")

    if request.sampling_params.repetition_penalty != 1.0:
        # groq supports frequency_penalty, but frequency_penalty and sampling_params.repetition_penalty
        # seem to have different semantics
        # frequency_penalty defaults to 0 is a float between -2.0 and 2.0
        # repetition_penalty defaults to 1 and is often set somewhere between 1.0 and 2.0
        # so we exclude it for now
        warnings.warn("repetition_penalty is not supported")

    if request.tool_config.tool_prompt_format != ToolPromptFormat.json:
        warnings.warn("tool_prompt_format is not used by Groq. Ignoring.")

    sampling_options = get_sampling_strategy_options(request.sampling_params)
    return CompletionCreateParams(
        model=request.model,
        messages=[_convert_message(message) for message in request.messages],
        logprobs=None,
        frequency_penalty=None,
        stream=request.stream,
        max_tokens=request.sampling_params.max_tokens or None,
        temperature=sampling_options.get("temperature", 1.0),
        top_p=sampling_options.get("top_p", 1.0),
        tools=[_convert_groq_tool_definition(tool) for tool in request.tools or []],
        tool_choice=(
            request.tool_config.tool_choice.value
            if request.tool_config.tool_choice
            else None
        ),
    )


def _convert_message(message: Message) -> ChatCompletionMessageParam:
    if message.role == "system":
        return ChatCompletionSystemMessageParam(role="system", content=message.content)
    elif message.role == "user":
        return ChatCompletionUserMessageParam(role="user", content=message.content)
    elif message.role == "assistant":
        return ChatCompletionAssistantMessageParam(
            role="assistant", content=message.content
        )
    else:
        raise ValueError(f"Invalid message role: {message.role}")


def _convert_groq_tool_definition(tool_definition: ToolDefinition) -> dict:
    # Groq requires a description for function tools
    if tool_definition.description is None:
        raise AssertionError("tool_definition.description is required")

    tool_parameters = tool_definition.parameters or {}
    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name=tool_definition.tool_name,
            description=tool_definition.description,
            parameters={
                key: _convert_groq_tool_parameter(param)
                for key, param in tool_parameters.items()
            },
        ),
    )


def _convert_groq_tool_parameter(tool_parameter: ToolParamDefinition) -> dict:
    param = {
        "type": tool_parameter.param_type,
    }
    if tool_parameter.description is not None:
        param["description"] = tool_parameter.description
    if tool_parameter.required is not None:
        param["required"] = tool_parameter.required
    if tool_parameter.default is not None:
        param["default"] = tool_parameter.default
    return param


def convert_chat_completion_response(
    response: ChatCompletion,
) -> ChatCompletionResponse:
    # groq only supports n=1 at time of writing, so there is only one choice
    choice = response.choices[0]
    if choice.finish_reason == "tool_calls":
        tool_calls = [
            _convert_groq_tool_call(tool_call)
            for tool_call in choice.message.tool_calls
        ]
        if any(isinstance(tool_call, UnparseableToolCall) for tool_call in tool_calls):
            # If we couldn't parse a tool call, jsonify the tool calls and return them
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    stop_reason=StopReason.end_of_message,
                    content=json.dumps(tool_calls, default=lambda x: x.model_dump()),
                ),
                logprobs=None,
            )
        else:
            # Otherwise, return tool calls as normal
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    tool_calls=tool_calls,
                    stop_reason=StopReason.end_of_message,
                    # Content is not optional
                    content="",
                ),
                logprobs=None,
            )
    else:
        return ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=choice.message.content,
                stop_reason=_map_finish_reason_to_stop_reason(choice.finish_reason),
            ),
        )


def _map_finish_reason_to_stop_reason(
    finish_reason: Literal["stop", "length", "tool_calls"],
) -> StopReason:
    """
    Convert a Groq chat completion finish_reason to a StopReason.

    finish_reason: Literal["stop", "length", "tool_calls"]
        - stop -> model hit a natural stop point or a provided stop sequence
        - length -> maximum number of tokens specified in the request was reached
        - tool_calls -> model called a tool
    """
    if finish_reason == "stop":
        return StopReason.end_of_turn
    elif finish_reason == "length":
        return StopReason.out_of_tokens
    elif finish_reason == "tool_calls":
        return StopReason.end_of_message
    else:
        raise ValueError(f"Invalid finish reason: {finish_reason}")


async def convert_chat_completion_response_stream(
    stream: Stream[ChatCompletionChunk],
) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
    event_type = ChatCompletionResponseEventType.start
    for chunk in stream:
        choice = chunk.choices[0]

        if choice.finish_reason:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=None,
                    stop_reason=_map_finish_reason_to_stop_reason(choice.finish_reason),
                )
            )
        elif choice.delta.tool_calls:
            # We assume there is only one tool call per chunk, but emit a warning in case we're wrong
            if len(choice.delta.tool_calls) > 1:
                warnings.warn(
                    "Groq returned multiple tool calls in one chunk. Using the first one, ignoring the rest."
                )

            # We assume Groq produces fully formed tool calls for each chunk
            tool_call = _convert_groq_tool_call(choice.delta.tool_calls[0])
            if isinstance(tool_call, ToolCall):
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=event_type,
                        delta=ToolCallDelta(
                            tool_call=tool_call,
                            parse_status=ToolCallParseStatus.succeeded,
                        ),
                    )
                )
            else:
                # Otherwise it's an UnparseableToolCall - return the raw tool call
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=event_type,
                        delta=ToolCallDelta(
                            tool_call=tool_call.model_dump_json(),
                            parse_status=ToolCallParseStatus.failed,
                        ),
                    )
                )
        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=event_type,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=None,
                )
            )
        event_type = ChatCompletionResponseEventType.progress


class UnparseableToolCall(BaseModel):
    """
    A ToolCall with arguments that are not valid JSON.
    Mirrors the ToolCall schema, but with arguments as a string.
    """

    call_id: str
    tool_name: str
    arguments: str


def _convert_groq_tool_call(
    tool_call: ChatCompletionMessageToolCall,
) -> Union[ToolCall, UnparseableToolCall]:
    """
    Convert a Groq tool call to a ToolCall.
    Returns an UnparseableToolCall if the tool call is not valid JSON.
    """
    try:
        arguments = json.loads(tool_call.function.arguments)
    except Exception as e:
        return UnparseableToolCall(
            call_id=tool_call.id,
            tool_name=tool_call.function.name,
            arguments=tool_call.function.arguments,
        )

    return ToolCall(
        call_id=tool_call.id,
        tool_name=tool_call.function.name,
        arguments=arguments,
    )
