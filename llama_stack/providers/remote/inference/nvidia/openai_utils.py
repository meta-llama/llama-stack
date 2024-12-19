# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import warnings
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from llama_models.llama3.api.datatypes import (
    BuiltinTool,
    # CompletionMessage,
    StopReason,
    # TokenLogProbs,
    ToolCall,
    ToolDefinition,
)
from llama_stack.apis.inference import CompletionMessage, TokenLogProbs
from openai import AsyncStream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam as OpenAIChatCompletionAssistantMessage,
    ChatCompletionChunk as OpenAIChatCompletionChunk,
    ChatCompletionMessageParam as OpenAIChatCompletionMessage,
    ChatCompletionMessageToolCallParam as OpenAIChatCompletionMessageToolCall,
    ChatCompletionSystemMessageParam as OpenAIChatCompletionSystemMessage,
    ChatCompletionToolMessageParam as OpenAIChatCompletionToolMessage,
    ChatCompletionUserMessageParam as OpenAIChatCompletionUserMessage,
)
from openai.types.chat.chat_completion import (
    Choice as OpenAIChoice,
    ChoiceLogprobs as OpenAIChoiceLogprobs,  # same as chat_completion_chunk ChoiceLogprobs
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as OpenAIFunction,
)
from openai.types.completion import Completion as OpenAICompletion
from openai.types.completion_choice import Logprobs as OpenAICompletionLogprobs

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    JsonSchemaResponseFormat,
    Message,
    SystemMessage,
    ToolCallDelta,
    ToolCallParseStatus,
    ToolResponseMessage,
    UserMessage,
)


def _convert_tooldef_to_openai_tool(tool: ToolDefinition) -> dict:
    """
    Convert a ToolDefinition to an OpenAI API-compatible dictionary.

    ToolDefinition:
        tool_name: str | BuiltinTool
        description: Optional[str]
        parameters: Optional[Dict[str, ToolParamDefinition]]

    ToolParamDefinition:
        param_type: str
        description: Optional[str]
        required: Optional[bool]
        default: Optional[Any]


    OpenAI spec -

    {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    param_name: {
                        "type": param_type,
                        "description": description,
                        "default": default,
                    },
                    ...
                },
                "required": [param_name, ...],
            },
        },
    }
    """
    out = {
        "type": "function",
        "function": {},
    }
    function = out["function"]

    if isinstance(tool.tool_name, BuiltinTool):
        function.update(name=tool.tool_name.value)  # TODO(mf): is this sufficient?
    else:
        function.update(name=tool.tool_name)

    if tool.description:
        function.update(description=tool.description)

    if tool.parameters:
        parameters = {
            "type": "object",
            "properties": {},
        }
        properties = parameters["properties"]
        required = []
        for param_name, param in tool.parameters.items():
            properties[param_name] = {"type": param.param_type}
            if param.description:
                properties[param_name].update(description=param.description)
            if param.default:
                properties[param_name].update(default=param.default)
            if param.required:
                required.append(param_name)

        if required:
            parameters.update(required=required)

        function.update(parameters=parameters)

    return out


def _convert_message(message: Message | Dict) -> OpenAIChatCompletionMessage:
    """
    Convert a Message to an OpenAI API-compatible dictionary.
    """
    # users can supply a dict instead of a Message object, we'll
    # convert it to a Message object and proceed with some type safety.
    if isinstance(message, dict):
        if "role" not in message:
            raise ValueError("role is required in message")
        if message["role"] == "user":
            message = UserMessage(**message)
        elif message["role"] == "assistant":
            message = CompletionMessage(**message)
        elif message["role"] == "ipython":
            message = ToolResponseMessage(**message)
        elif message["role"] == "system":
            message = SystemMessage(**message)
        else:
            raise ValueError(f"Unsupported message role: {message['role']}")

    out: OpenAIChatCompletionMessage = None
    if isinstance(message, UserMessage):
        out = OpenAIChatCompletionUserMessage(
            role="user",
            content=message.content,  # TODO(mf): handle image content
        )
    elif isinstance(message, CompletionMessage):
        out = OpenAIChatCompletionAssistantMessage(
            role="assistant",
            content=message.content,
            tool_calls=[
                OpenAIChatCompletionMessageToolCall(
                    id=tool.call_id,
                    function=OpenAIFunction(
                        name=tool.tool_name,
                        arguments=json.dumps(tool.arguments),
                    ),
                    type="function",
                )
                for tool in message.tool_calls
            ],
        )
    elif isinstance(message, ToolResponseMessage):
        out = OpenAIChatCompletionToolMessage(
            role="tool",
            tool_call_id=message.call_id,
            content=message.content,
        )
    elif isinstance(message, SystemMessage):
        out = OpenAIChatCompletionSystemMessage(
            role="system",
            content=message.content,
        )
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

    return out


def convert_chat_completion_request(
    request: ChatCompletionRequest,
    n: int = 1,
) -> dict:
    """
    Convert a ChatCompletionRequest to an OpenAI API-compatible dictionary.
    """
    # model -> model
    # messages -> messages
    # sampling_params  TODO(mattf): review strategy
    #  strategy=greedy -> nvext.top_k = -1, temperature = temperature
    #  strategy=top_p -> nvext.top_k = -1, top_p = top_p
    #  strategy=top_k -> nvext.top_k = top_k
    #  temperature -> temperature
    #  top_p -> top_p
    #  top_k -> nvext.top_k
    #  max_tokens -> max_tokens
    #  repetition_penalty -> nvext.repetition_penalty
    # response_format -> GrammarResponseFormat TODO(mf)
    # response_format -> JsonSchemaResponseFormat: response_format = "json_object" & nvext["guided_json"] = json_schema
    # tools -> tools
    # tool_choice ("auto", "required") -> tool_choice
    # tool_prompt_format -> TBD
    # stream -> stream
    # logprobs -> logprobs

    if request.response_format and not isinstance(
        request.response_format, JsonSchemaResponseFormat
    ):
        raise ValueError(
            f"Unsupported response format: {request.response_format}. "
            "Only JsonSchemaResponseFormat is supported."
        )

    nvext = {}
    payload: Dict[str, Any] = dict(
        model=request.model,
        messages=[_convert_message(message) for message in request.messages],
        stream=request.stream,
        n=n,
        extra_body=dict(nvext=nvext),
        extra_headers={
            b"User-Agent": b"llama-stack: nvidia-inference-adapter",
        },
    )

    if request.response_format:
        # server bug - setting guided_json changes the behavior of response_format resulting in an error
        # payload.update(response_format="json_object")
        nvext.update(guided_json=request.response_format.json_schema)

    if request.tools:
        payload.update(
            tools=[_convert_tooldef_to_openai_tool(tool) for tool in request.tools]
        )
        if request.tool_choice:
            payload.update(
                tool_choice=request.tool_choice.value
            )  # we cannot include tool_choice w/o tools, server will complain

    if request.logprobs:
        payload.update(logprobs=True)
        payload.update(top_logprobs=request.logprobs.top_k)

    if request.sampling_params:
        nvext.update(repetition_penalty=request.sampling_params.repetition_penalty)

        if request.sampling_params.max_tokens:
            payload.update(max_tokens=request.sampling_params.max_tokens)

        if request.sampling_params.strategy == "top_p":
            nvext.update(top_k=-1)
            payload.update(top_p=request.sampling_params.top_p)
        elif request.sampling_params.strategy == "top_k":
            if (
                request.sampling_params.top_k != -1
                and request.sampling_params.top_k < 1
            ):
                warnings.warn("top_k must be -1 or >= 1")
            nvext.update(top_k=request.sampling_params.top_k)
        elif request.sampling_params.strategy == "greedy":
            nvext.update(top_k=-1)
            payload.update(temperature=request.sampling_params.temperature)

    return payload


def _convert_openai_finish_reason(finish_reason: str) -> StopReason:
    """
    Convert an OpenAI chat completion finish_reason to a StopReason.

    finish_reason: Literal["stop", "length", "tool_calls", ...]
        - stop: model hit a natural stop point or a provided stop sequence
        - length: maximum number of tokens specified in the request was reached
        - tool_calls: model called a tool

    ->

    class StopReason(Enum):
        end_of_turn = "end_of_turn"
        end_of_message = "end_of_message"
        out_of_tokens = "out_of_tokens"
    """

    # TODO(mf): are end_of_turn and end_of_message semantics correct?
    return {
        "stop": StopReason.end_of_turn,
        "length": StopReason.out_of_tokens,
        "tool_calls": StopReason.end_of_message,
    }.get(finish_reason, StopReason.end_of_turn)


def _convert_openai_tool_calls(
    tool_calls: List[OpenAIChatCompletionMessageToolCall],
) -> List[ToolCall]:
    """
    Convert an OpenAI ChatCompletionMessageToolCall list into a list of ToolCall.

    OpenAI ChatCompletionMessageToolCall:
        id: str
        function: Function
        type: Literal["function"]

    OpenAI Function:
        arguments: str
        name: str

    ->

    ToolCall:
        call_id: str
        tool_name: str
        arguments: Dict[str, ...]
    """
    if not tool_calls:
        return []  # CompletionMessage tool_calls is not optional

    return [
        ToolCall(
            call_id=call.id,
            tool_name=call.function.name,
            arguments=json.loads(call.function.arguments),
        )
        for call in tool_calls
    ]


def _convert_openai_logprobs(
    logprobs: OpenAIChoiceLogprobs,
) -> Optional[List[Any]]:
    """
    Convert an OpenAI ChoiceLogprobs into a list of TokenLogProbs.

    OpenAI ChoiceLogprobs:
        content: Optional[List[ChatCompletionTokenLogprob]]

    OpenAI ChatCompletionTokenLogprob:
        token: str
        logprob: float
        top_logprobs: List[TopLogprob]

    OpenAI TopLogprob:
        token: str
        logprob: float

    ->

    TokenLogProbs:
        logprobs_by_token: Dict[str, float]
         - token, logprob

    """
    if not logprobs:
        return None

    return [
        TokenLogProbs(
            logprobs_by_token={
                logprobs.token: logprobs.logprob for logprobs in content.top_logprobs
            }
        )
        for content in logprobs.content
    ]


def convert_openai_chat_completion_choice(
    choice: OpenAIChoice,
) -> ChatCompletionResponse:
    """
    Convert an OpenAI Choice into a ChatCompletionResponse.

    OpenAI Choice:
        message: ChatCompletionMessage
        finish_reason: str
        logprobs: Optional[ChoiceLogprobs]

    OpenAI ChatCompletionMessage:
        role: Literal["assistant"]
        content: Optional[str]
        tool_calls: Optional[List[ChatCompletionMessageToolCall]]

    ->

    ChatCompletionResponse:
        completion_message: CompletionMessage
        logprobs: Optional[List[TokenLogProbs]]

    CompletionMessage:
        role: Literal["assistant"]
        content: str | ImageMedia | List[str | ImageMedia]
        stop_reason: StopReason
        tool_calls: List[ToolCall]

    class StopReason(Enum):
        end_of_turn = "end_of_turn"
        end_of_message = "end_of_message"
        out_of_tokens = "out_of_tokens"
    """
    assert (
        hasattr(choice, "message") and choice.message
    ), "error in server response: message not found"
    assert (
        hasattr(choice, "finish_reason") and choice.finish_reason
    ), "error in server response: finish_reason not found"

    return ChatCompletionResponse(
        completion_message=CompletionMessage(
            content=choice.message.content
            or "",  # CompletionMessage content is not optional
            stop_reason=_convert_openai_finish_reason(choice.finish_reason),
            tool_calls=_convert_openai_tool_calls(choice.message.tool_calls),
        ),
        logprobs=_convert_openai_logprobs(choice.logprobs),
    )


async def convert_openai_chat_completion_stream(
    stream: AsyncStream[OpenAIChatCompletionChunk],
) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
    """
    Convert a stream of OpenAI chat completion chunks into a stream
    of ChatCompletionResponseStreamChunk.

    OpenAI ChatCompletionChunk:
        choices: List[Choice]

    OpenAI Choice:  # different from the non-streamed Choice
        delta: ChoiceDelta
        finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]]
        logprobs: Optional[ChoiceLogprobs]

    OpenAI ChoiceDelta:
        content: Optional[str]
        role: Optional[Literal["system", "user", "assistant", "tool"]]
        tool_calls: Optional[List[ChoiceDeltaToolCall]]

    OpenAI ChoiceDeltaToolCall:
        index: int
        id: Optional[str]
        function: Optional[ChoiceDeltaToolCallFunction]
        type: Optional[Literal["function"]]

    OpenAI ChoiceDeltaToolCallFunction:
        name: Optional[str]
        arguments: Optional[str]

    ->

    ChatCompletionResponseStreamChunk:
        event: ChatCompletionResponseEvent

    ChatCompletionResponseEvent:
        event_type: ChatCompletionResponseEventType
        delta: Union[str, ToolCallDelta]
        logprobs: Optional[List[TokenLogProbs]]
        stop_reason: Optional[StopReason]

    ChatCompletionResponseEventType:
        start = "start"
        progress = "progress"
        complete = "complete"

    ToolCallDelta:
        content: Union[str, ToolCall]
        parse_status: ToolCallParseStatus

    ToolCall:
        call_id: str
        tool_name: str
        arguments: str

    ToolCallParseStatus:
        started = "started"
        in_progress = "in_progress"
        failure = "failure"
        success = "success"

    TokenLogProbs:
        logprobs_by_token: Dict[str, float]
         - token, logprob

    StopReason:
        end_of_turn = "end_of_turn"
        end_of_message = "end_of_message"
        out_of_tokens = "out_of_tokens"
    """

    # generate a stream of ChatCompletionResponseEventType: start -> progress -> progress -> ...
    def _event_type_generator() -> (
        Generator[ChatCompletionResponseEventType, None, None]
    ):
        yield ChatCompletionResponseEventType.start
        while True:
            yield ChatCompletionResponseEventType.progress

    event_type = _event_type_generator()

    # we implement NIM specific semantics, the main difference from OpenAI
    # is that tool_calls are always produced as a complete call. there is no
    # intermediate / partial tool call streamed. because of this, we can
    # simplify the logic and not concern outselves with parse_status of
    # started/in_progress/failed. we can always assume success.
    #
    # a stream of ChatCompletionResponseStreamChunk consists of
    #  0. a start event
    #  1. zero or more progress events
    #   - each progress event has a delta
    #   - each progress event may have a stop_reason
    #   - each progress event may have logprobs
    #   - each progress event may have tool_calls
    #     if a progress event has tool_calls,
    #      it is fully formed and
    #      can be emitted with a parse_status of success
    #  2. a complete event

    stop_reason = None

    async for chunk in stream:
        choice = chunk.choices[0]  # assuming only one choice per chunk

        # we assume there's only one finish_reason in the stream
        stop_reason = _convert_openai_finish_reason(choice.finish_reason) or stop_reason

        # if there's a tool call, emit an event for each tool in the list
        # if tool call and content, emit both separately

        if choice.delta.tool_calls:
            # the call may have content and a tool call. ChatCompletionResponseEvent
            # does not support both, so we emit the content first
            if choice.delta.content:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=next(event_type),
                        delta=choice.delta.content,
                        logprobs=_convert_openai_logprobs(choice.logprobs),
                    )
                )

            # it is possible to have parallel tool calls in stream, but
            # ChatCompletionResponseEvent only supports one per stream
            if len(choice.delta.tool_calls) > 1:
                warnings.warn(
                    "multiple tool calls found in a single delta, using the first, ignoring the rest"
                )

            # NIM only produces fully formed tool calls, so we can assume success
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=next(event_type),
                    delta=ToolCallDelta(
                        content=_convert_openai_tool_calls(choice.delta.tool_calls)[0],
                        parse_status=ToolCallParseStatus.success,
                    ),
                    logprobs=_convert_openai_logprobs(choice.logprobs),
                )
            )
        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=next(event_type),
                    delta=choice.delta.content or "",  # content is not optional
                    logprobs=_convert_openai_logprobs(choice.logprobs),
                )
            )

    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.complete,
            delta="",
            stop_reason=stop_reason,
        )
    )


def convert_completion_request(
    request: CompletionRequest,
    n: int = 1,
) -> dict:
    """
    Convert a ChatCompletionRequest to an OpenAI API-compatible dictionary.
    """
    # model -> model
    # prompt -> prompt
    # sampling_params  TODO(mattf): review strategy
    #  strategy=greedy -> nvext.top_k = -1, temperature = temperature
    #  strategy=top_p -> nvext.top_k = -1, top_p = top_p
    #  strategy=top_k -> nvext.top_k = top_k
    #  temperature -> temperature
    #  top_p -> top_p
    #  top_k -> nvext.top_k
    #  max_tokens -> max_tokens
    #  repetition_penalty -> nvext.repetition_penalty
    # response_format -> nvext.guided_json
    # stream -> stream
    # logprobs.top_k -> logprobs

    nvext = {}
    payload: Dict[str, Any] = dict(
        model=request.model,
        prompt=request.content,
        stream=request.stream,
        extra_body=dict(nvext=nvext),
        extra_headers={
            b"User-Agent": b"llama-stack: nvidia-inference-adapter",
        },
        n=n,
    )

    if request.response_format:
        # this is not openai compliant, it is a nim extension
        nvext.update(guided_json=request.response_format.json_schema)

    if request.logprobs:
        payload.update(logprobs=request.logprobs.top_k)

    if request.sampling_params:
        nvext.update(repetition_penalty=request.sampling_params.repetition_penalty)

        if request.sampling_params.max_tokens:
            payload.update(max_tokens=request.sampling_params.max_tokens)

        if request.sampling_params.strategy == "top_p":
            nvext.update(top_k=-1)
            payload.update(top_p=request.sampling_params.top_p)
        elif request.sampling_params.strategy == "top_k":
            if (
                request.sampling_params.top_k != -1
                and request.sampling_params.top_k < 1
            ):
                warnings.warn("top_k must be -1 or >= 1")
            nvext.update(top_k=request.sampling_params.top_k)
        elif request.sampling_params.strategy == "greedy":
            nvext.update(top_k=-1)
            payload.update(temperature=request.sampling_params.temperature)

    return payload


def _convert_openai_completion_logprobs(
    logprobs: Optional[OpenAICompletionLogprobs],
) -> Optional[List[TokenLogProbs]]:
    """
    Convert an OpenAI CompletionLogprobs into a list of TokenLogProbs.

    OpenAI CompletionLogprobs:
        text_offset: Optional[List[int]]
        token_logprobs: Optional[List[float]]
        tokens: Optional[List[str]]
        top_logprobs: Optional[List[Dict[str, float]]]

    ->

    TokenLogProbs:
        logprobs_by_token: Dict[str, float]
            - token, logprob
    """
    if not logprobs:
        return None

    return [
        TokenLogProbs(logprobs_by_token=logprobs) for logprobs in logprobs.top_logprobs
    ]


def convert_openai_completion_choice(
    choice: OpenAIChoice,
) -> CompletionResponse:
    """
    Convert an OpenAI Completion Choice into a CompletionResponse.

    OpenAI Completion Choice:
        text: str
        finish_reason: str
        logprobs: Optional[ChoiceLogprobs]

    ->

    CompletionResponse:
        completion_message: CompletionMessage
        logprobs: Optional[List[TokenLogProbs]]

    CompletionMessage:
        role: Literal["assistant"]
        content: str | ImageMedia | List[str | ImageMedia]
        stop_reason: StopReason
        tool_calls: List[ToolCall]

    class StopReason(Enum):
        end_of_turn = "end_of_turn"
        end_of_message = "end_of_message"
        out_of_tokens = "out_of_tokens"
    """
    return CompletionResponse(
        content=choice.text,
        stop_reason=_convert_openai_finish_reason(choice.finish_reason),
        logprobs=_convert_openai_completion_logprobs(choice.logprobs),
    )


async def convert_openai_completion_stream(
    stream: AsyncStream[OpenAICompletion],
) -> AsyncGenerator[CompletionResponse, None]:
    """
    Convert a stream of OpenAI Completions into a stream
    of ChatCompletionResponseStreamChunks.

    OpenAI Completion:
        id: str
        choices: List[OpenAICompletionChoice]
        created: int
        model: str
        system_fingerprint: Optional[str]
        usage: Optional[OpenAICompletionUsage]

    OpenAI CompletionChoice:
        finish_reason: str
        index: int
        logprobs: Optional[OpenAILogprobs]
        text: str

    ->

    CompletionResponseStreamChunk:
        delta: str
        stop_reason: Optional[StopReason]
        logprobs: Optional[List[TokenLogProbs]]
    """
    async for chunk in stream:
        choice = chunk.choices[0]
        yield CompletionResponseStreamChunk(
            delta=choice.text,
            stop_reason=_convert_openai_finish_reason(choice.finish_reason),
            logprobs=_convert_openai_completion_logprobs(choice.logprobs),
        )
