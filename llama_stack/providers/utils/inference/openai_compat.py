# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import logging
import warnings
from typing import AsyncGenerator, Dict, Generator, Iterable, List, Optional, Union

from llama_stack.apis.common.content_types import (
    ImageContentItem,
    InterleavedContent,
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
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    JsonSchemaResponseFormat,
    Message,
    SystemMessage,
    TokenLogProbs,
    ToolChoice,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.models.llama.datatypes import (
    BuiltinTool,
    GreedySamplingStrategy,
    SamplingParams,
    StopReason,
    ToolCall,
    ToolDefinition,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    convert_image_content_to_url,
    decode_assistant_message,
)

from openai import AsyncStream, Stream
from openai.types.chat import (
    ChatCompletionAssistantMessageParam as OpenAIChatCompletionAssistantMessage,
    ChatCompletionChunk as OpenAIChatCompletionChunk,
    ChatCompletionContentPartImageParam as OpenAIChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam as OpenAIChatCompletionContentPartTextParam,
    ChatCompletionMessageParam as OpenAIChatCompletionMessage,
    ChatCompletionMessageToolCall as OpenAIChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam as OpenAIChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam as OpenAIChatCompletionSystemMessage,
    ChatCompletionToolMessageParam as OpenAIChatCompletionToolMessage,
    ChatCompletionUserMessageParam as OpenAIChatCompletionUserMessage,
)
from openai.types.chat.chat_completion import (
    Choice as OpenAIChoice,
    ChoiceLogprobs as OpenAIChoiceLogprobs,  # same as chat_completion_chunk ChoiceLogprobs
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ImageURL as OpenAIImageURL,
)
from openai.types.chat.chat_completion_message_tool_call_param import (
    Function as OpenAIFunction,
)
from pydantic import BaseModel

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

        if params.stop is not None:
            options["stop"] = params.stop

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
            for token, token_lp in zip(
                logprobs.tokens, logprobs.token_logprobs, strict=False
            )
        ]
    return None


def convert_openai_completion_logprobs_stream(
    text: str, logprobs: Optional[Union[float, OpenAICompatLogprobs]]
):
    if logprobs is None:
        return None
    if isinstance(logprobs, float):
        # Adapt response from Together CompletionChoicesChunk
        return [TokenLogProbs(logprobs_by_token={text: logprobs})]
    if hasattr(logprobs, "top_logprobs"):
        return [TokenLogProbs(logprobs_by_token=x) for x in logprobs.top_logprobs]
    return None


def process_completion_response(
    response: OpenAICompatCompletionResponse,
) -> CompletionResponse:
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
    if choice.finish_reason == "tool_calls":
        if not choice.message or not choice.message.tool_calls:
            raise ValueError("Tool calls are not present in the response")

        tool_calls = [
            convert_tool_call(tool_call) for tool_call in choice.message.tool_calls
        ]
        if any(isinstance(tool_call, UnparseableToolCall) for tool_call in tool_calls):
            # If we couldn't parse a tool call, jsonify the tool calls and return them
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    stop_reason=StopReason.end_of_turn,
                    content=json.dumps(tool_calls, default=lambda x: x.model_dump()),
                ),
                logprobs=None,
            )
        else:
            # Otherwise, return tool calls as normal
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    tool_calls=tool_calls,
                    stop_reason=StopReason.end_of_turn,
                    # Content is not optional
                    content="",
                ),
                logprobs=None,
            )

    # TODO: This does not work well with tool calls for vLLM remote provider
    #   Ref: https://github.com/meta-llama/llama-stack/issues/1058
    raw_message = decode_assistant_message(
        text_from_choice(choice), get_stop_reason(choice.finish_reason)
    )

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


async def convert_message_to_openai_dict(
    message: Message, download: bool = False
) -> dict:
    async def _convert_content(content) -> dict:
        if isinstance(content, ImageContentItem):
            return {
                "type": "image_url",
                "image_url": {
                    "url": await convert_image_content_to_url(
                        content, download=download
                    ),
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


async def convert_message_to_openai_dict_new(
    message: Message | Dict,
) -> OpenAIChatCompletionMessage:
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
        elif message["role"] == "tool":
            message = ToolResponseMessage(**message)
        elif message["role"] == "system":
            message = SystemMessage(**message)
        else:
            raise ValueError(f"Unsupported message role: {message['role']}")

    # Map Llama Stack spec to OpenAI spec -
    #  str -> str
    #  {"type": "text", "text": ...} -> {"type": "text", "text": ...}
    #  {"type": "image", "image": {"url": {"uri": ...}}} -> {"type": "image_url", "image_url": {"url": ...}}
    #  {"type": "image", "image": {"data": ...}} -> {"type": "image_url", "image_url": {"url": "data:image/?;base64,..."}}
    #  List[...] -> List[...]
    async def _convert_message_content(
        content: InterleavedContent,
    ) -> Union[str, Iterable[OpenAIChatCompletionContentPartParam]]:
        async def impl(
            content_: InterleavedContent,
        ) -> Union[
            str,
            OpenAIChatCompletionContentPartParam,
            List[OpenAIChatCompletionContentPartParam],
        ]:
            # Llama Stack and OpenAI spec match for str and text input
            if isinstance(content_, str):
                return content_
            elif isinstance(content_, TextContentItem):
                return OpenAIChatCompletionContentPartTextParam(
                    type="text",
                    text=content_.text,
                )
            elif isinstance(content_, ImageContentItem):
                return OpenAIChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=OpenAIImageURL(
                        url=await convert_image_content_to_url(content_)
                    ),
                )
            elif isinstance(content_, list):
                return [await impl(item) for item in content_]
            else:
                raise ValueError(f"Unsupported content type: {type(content_)}")

        ret = await impl(content)

        # OpenAI*Message expects a str or list
        if isinstance(ret, str) or isinstance(ret, list):
            return ret
        else:
            return [ret]

    out: OpenAIChatCompletionMessage = None
    if isinstance(message, UserMessage):
        out = OpenAIChatCompletionUserMessage(
            role="user",
            content=await _convert_message_content(message.content),
        )
    elif isinstance(message, CompletionMessage):
        out = OpenAIChatCompletionAssistantMessage(
            role="assistant",
            content=await _convert_message_content(message.content),
            tool_calls=[
                # OpenAIChatCompletionMessageToolCall(
                #     id=tool.call_id,
                #     function=OpenAIFunction(
                #         name=(
                #             tool.tool_name
                #             if not isinstance(tool.tool_name, BuiltinTool)
                #             else tool.tool_name.value
                #         ),
                #         arguments=json.dumps(tool.arguments),
                #     ),
                #     type="function",
                # )
                # using a dict instead of OpenAIChatCompletionMessageToolCall object
                # as it fails to get json encoded
                {
                    "id": tool.call_id,
                    "function": {
                        "name": (
                            tool.tool_name
                            if not isinstance(tool.tool_name, BuiltinTool)
                            else tool.tool_name.value
                        ),
                        "arguments": json.dumps(tool.arguments),
                    },
                    "type": "function",
                }
                for tool in message.tool_calls
            ]
            or None,
        )
    elif isinstance(message, ToolResponseMessage):
        out = OpenAIChatCompletionToolMessage(
            role="tool",
            tool_call_id=message.call_id,
            content=await _convert_message_content(message.content),
        )
    elif isinstance(message, SystemMessage):
        out = OpenAIChatCompletionSystemMessage(
            role="system",
            content=await _convert_message_content(message.content),
        )
    else:
        raise ValueError(f"Unsupported message type: {type(message)}")

    return out


def convert_tool_call(
    tool_call: OpenAIChatCompletionMessageToolCall,
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
            arguments_json=tool_call.function.arguments,
        )
    except Exception:
        return UnparseableToolCall(
            call_id=tool_call.id or "",
            tool_name=tool_call.function.name or "",
            arguments=tool_call.function.arguments or "",
        )

    return valid_tool_call


PYTHON_TYPE_TO_LITELLM_TYPE = {
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "str": "string",
}


def convert_tooldef_to_openai_tool(tool: ToolDefinition) -> dict:
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
            properties[param_name] = {
                "type": PYTHON_TYPE_TO_LITELLM_TYPE.get(
                    param.param_type, param.param_type
                )
            }
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

    ls_tool_calls = []
    for call in tool_calls:
        args = call.function.arguments
        # TGI is sending a dict instead of a json string
        # While OpenAI spec expects a json string
        if isinstance(args, str):
            arguments = json.loads(args)
            arguments_json = args
        elif isinstance(args, dict):
            arguments = args
            arguments_json = json.dumps(args)
        else:
            raise ValueError(f"Unsupported arguments type: {type(args)}")

        ls_tool_calls.append(
            ToolCall(
                call_id=call.id,
                tool_name=call.function.name,
                arguments=arguments,
                arguments_json=arguments_json,
            )
        )

    return ls_tool_calls


def _convert_openai_logprobs(
    logprobs: OpenAIChoiceLogprobs,
) -> Optional[List[TokenLogProbs]]:
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
        logprobs=_convert_openai_logprobs(getattr(choice, "logprobs", None)),
    )


async def convert_openai_chat_completion_stream(
    stream: Union[
        AsyncStream[OpenAIChatCompletionChunk], Stream[OpenAIChatCompletionChunk]
    ],
    enable_incremental_tool_calls: bool,
) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
    """
    Convert a stream of OpenAI chat completion chunks into a stream
    of ChatCompletionResponseStreamChunk.
    """

    async def yield_from_stream(stream):
        if isinstance(stream, AsyncGenerator):
            async for chunk in stream:
                yield chunk
        elif isinstance(stream, Generator):
            for chunk in stream:
                yield chunk

    yield ChatCompletionResponseStreamChunk(
        event=ChatCompletionResponseEvent(
            event_type=ChatCompletionResponseEventType.start,
            delta=TextDelta(text=""),
        )
    )
    event_type = ChatCompletionResponseEventType.progress

    stop_reason = None
    tool_call_idx_to_buffer = {}

    async for chunk in yield_from_stream(stream):
        choice = chunk.choices[0]  # assuming only one choice per chunk

        # we assume there's only one finish_reason in the stream
        stop_reason = _convert_openai_finish_reason(choice.finish_reason) or stop_reason
        logprobs = getattr(choice, "logprobs", None)

        # if there's a tool call, emit an event for each tool in the list
        # if tool call and content, emit both separately
        if choice.delta.tool_calls:
            # the call may have content and a tool call. ChatCompletionResponseEvent
            # does not support both, so we emit the content first
            if choice.delta.content:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=event_type,
                        delta=TextDelta(text=choice.delta.content),
                        logprobs=_convert_openai_logprobs(logprobs),
                    )
                )

            # it is possible to have parallel tool calls in stream, but
            # ChatCompletionResponseEvent only supports one per stream
            if len(choice.delta.tool_calls) > 1:
                warnings.warn(
                    "multiple tool calls found in a single delta, using the first, ignoring the rest",
                    stacklevel=2,
                )

            if not enable_incremental_tool_calls:
                for tool_call in choice.delta.tool_calls:
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=event_type,
                            delta=ToolCallDelta(
                                tool_call=_convert_openai_tool_calls([tool_call])[0],
                                parse_status=ToolCallParseStatus.succeeded,
                            ),
                            logprobs=_convert_openai_logprobs(logprobs),
                        )
                    )
            else:
                if isinstance(choice.delta.tool_calls, list):
                    tool_calls = choice.delta.tool_calls
                    for tool_call in tool_calls:
                        idx = tool_call.index if hasattr(tool_call, "index") else 0

                        if idx not in tool_call_idx_to_buffer:
                            tool_call_idx_to_buffer[idx] = {
                                "call_id": tool_call.id,
                                "name": None,
                                "arguments": "",
                                "content": "",
                            }

                        buffer = tool_call_idx_to_buffer[idx]

                        if tool_call.function:
                            if tool_call.function.name:
                                buffer["name"] = tool_call.function.name
                                delta = f"{buffer['name']}("
                                buffer["content"] += delta

                            if tool_call.function.arguments:
                                delta = tool_call.function.arguments
                                buffer["arguments"] += delta
                                buffer["content"] += delta

                            yield ChatCompletionResponseStreamChunk(
                                event=ChatCompletionResponseEvent(
                                    event_type=event_type,
                                    delta=ToolCallDelta(
                                        tool_call=delta,
                                        parse_status=ToolCallParseStatus.in_progress,
                                    ),
                                    logprobs=_convert_openai_logprobs(logprobs),
                                )
                            )
                # TGI streams a non-openai compat response
                elif isinstance(choice.delta.tool_calls, dict):
                    # tool_calls is a dict of the format
                    # {
                    #   'index': 0,
                    #   'id': '',
                    #   'type': 'function',
                    #   'function': {
                    #       'name': None,
                    #       'arguments': '{"'
                    #   }
                    # }
                    tool_call = choice.delta.tool_calls
                    idx = tool_call["index"] if "index" in tool_call else 0

                    if idx not in tool_call_idx_to_buffer:
                        tool_call_idx_to_buffer[idx] = {
                            "call_id": tool_call["id"],
                            "name": None,
                            "arguments": "",
                            "content": "",
                        }

                    buffer = tool_call_idx_to_buffer[idx]

                    if "function" in tool_call:
                        function = tool_call["function"]
                        if function["name"]:
                            buffer["name"] = function["name"]
                            delta = f"{buffer['name']}("
                            buffer["content"] += delta

                        if function["arguments"]:
                            delta = function["arguments"]
                            buffer["arguments"] += delta
                            buffer["content"] += delta

                        yield ChatCompletionResponseStreamChunk(
                            event=ChatCompletionResponseEvent(
                                event_type=event_type,
                                delta=ToolCallDelta(
                                    tool_call=delta,
                                    parse_status=ToolCallParseStatus.in_progress,
                                ),
                                logprobs=_convert_openai_logprobs(logprobs),
                            )
                        )
        elif choice.delta.content:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=event_type,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=_convert_openai_logprobs(logprobs),
                )
            )

    for idx, buffer in tool_call_idx_to_buffer.items():
        logger.debug(f"toolcall_buffer[{idx}]: {buffer}")
        if buffer["name"]:
            delta = ")"
            buffer["content"] += delta
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=event_type,
                    delta=ToolCallDelta(
                        tool_call=delta,
                        parse_status=ToolCallParseStatus.in_progress,
                    ),
                    logprobs=None,
                )
            )

            try:
                arguments = json.loads(buffer["arguments"])
                tool_call = ToolCall(
                    call_id=buffer["call_id"],
                    tool_name=buffer["name"],
                    arguments=arguments,
                    arguments_json=buffer["arguments"],
                )
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
            except json.JSONDecodeError as e:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call=buffer["content"],
                            parse_status=ToolCallParseStatus.failed,
                        ),
                        stop_reason=stop_reason,
                    )
                )
        else:
            # name is None but we have the arguments contain the entire function call
            # example response where arguments is -->
            # '{"function": {"_name": "get_weather", "location": "San Francisco, CA"}}<|eot_id|>'
            # - parse the arguments
            # - build try to build ToolCall and return it or return the content as is

            if buffer["arguments"]:
                arguments = buffer["arguments"]
                # remove the eot_id and eom_id from the arguments
                if arguments.endswith("<|eom_id|>"):
                    arguments = arguments[: -len("<|eom_id|>")]
                if arguments.endswith("<|eot_id|>"):
                    arguments = arguments[: -len("<|eot_id|>")]

                arguments = json.loads(arguments)
                try:
                    tool_name = arguments["function"].pop("_name", None)
                    parsed_tool_call = ToolCall(
                        call_id=buffer["call_id"],
                        tool_name=tool_name,
                        arguments=arguments["function"],
                        arguments_json=json.dumps(arguments["function"]),
                    )
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                tool_call=parsed_tool_call,
                                parse_status=ToolCallParseStatus.succeeded,
                            ),
                            stop_reason=stop_reason,
                        )
                    )
                except (KeyError, json.JSONDecodeError) as e:
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=ToolCallDelta(
                                tool_call=buffer["content"],
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


async def convert_completion_request_to_openai_params(
    request: CompletionRequest,
) -> dict:
    """
    Convert a ChatCompletionRequest to an OpenAI API-compatible dictionary.
    """
    input_dict = {}
    if request.logprobs:
        input_dict["logprobs"] = request.logprobs.top_k

    return {
        "model": request.model,
        "prompt": request.content,
        **input_dict,
        "stream": request.stream,
        **get_sampling_options(request.sampling_params),
        "n": 1,
    }


async def convert_chat_completion_request_to_openai_params(
    request: ChatCompletionRequest,
) -> dict:
    """
    Convert a ChatCompletionRequest to an OpenAI chat completion request.
    """
    input_dict = {}

    input_dict["messages"] = [
        await convert_message_to_openai_dict_new(m) for m in request.messages
    ]
    if fmt := request.response_format:
        if not isinstance(fmt, JsonSchemaResponseFormat):
            raise ValueError(
                f"Unsupported response format: {type(fmt)}. Only JsonSchemaResponseFormat is supported."
            )

        fmt = fmt.json_schema
        name = fmt["title"]
        del fmt["title"]
        fmt["additionalProperties"] = False

        # Apply additionalProperties: False recursively to all objects
        fmt = _add_additional_properties_recursive(fmt)

        from rich.pretty import pprint

        pprint(fmt)

        input_dict["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": fmt,
                "strict": True,
            },
        }

    if request.tools:
        input_dict["tools"] = [
            convert_tooldef_to_openai_tool(tool) for tool in request.tools
        ]
        if request.tool_config.tool_choice:
            input_dict["tool_choice"] = (
                request.tool_config.tool_choice.value
                if isinstance(request.tool_config.tool_choice, ToolChoice)
                else request.tool_config.tool_choice
            )

    return {
        "model": request.model,
        **input_dict,
        "stream": request.stream,
        **get_sampling_options(request.sampling_params),
        "n": 1,
    }


def _add_additional_properties_recursive(schema):
    """
    Recursively add `additionalProperties: False` to all object schemas
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False

            # Add required field with all property keys if properties exist
            if "properties" in schema and schema["properties"]:
                schema["required"] = list(schema["properties"].keys())

        if "properties" in schema:
            for prop_schema in schema["properties"].values():
                _add_additional_properties_recursive(prop_schema)

        for key in ["anyOf", "allOf", "oneOf"]:
            if key in schema:
                for sub_schema in schema[key]:
                    _add_additional_properties_recursive(sub_schema)

        if "not" in schema:
            _add_additional_properties_recursive(schema["not"])

        # Handle $defs/$ref
        if "$defs" in schema:
            for def_schema in schema["$defs"].values():
                _add_additional_properties_recursive(def_schema)

    return schema
