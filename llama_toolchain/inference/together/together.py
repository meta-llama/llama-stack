# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import AsyncGenerator, Dict

from llama_models.llama3.api.datatypes import (
    BuiltinTool,
    CompletionMessage,
    Message,
    StopReason,
    ToolCall,
)
from llama_models.llama3.api.tool_utils import ToolUtils
from llama_models.sku_list import resolve_model
from together import Together

from llama_toolchain.distribution.datatypes import Api, ProviderSpec
from llama_toolchain.inference.api import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    Inference,
    ToolCallDelta,
    ToolCallParseStatus,
)

from .config import TogetherImplConfig

TOGETHER_SUPPORTED_MODELS = {
    "Meta-Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Meta-Llama3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "Meta-Llama3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
}


async def get_provider_impl(
    config: TogetherImplConfig, _deps: Dict[Api, ProviderSpec]
) -> Inference:
    assert isinstance(
        config, TogetherImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = TogetherInference(config)
    await impl.initialize()
    return impl


class TogetherInference(Inference):
    def __init__(self, config: TogetherImplConfig) -> None:
        self.config = config

    @property
    def client(self) -> Together:
        return Together(api_key=self.config.api_key)

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    def _messages_to_together_messages(self, messages: list[Message]) -> list:
        together_messages = []
        for message in messages:
            if message.role == "ipython":
                role = "tool"
            else:
                role = message.role
            together_messages.append({"role": role, "content": message.content})

        return together_messages

    def resolve_together_model(self, model_name: str) -> str:
        model = resolve_model(model_name)
        assert (
            model is not None
            and model.descriptor(shorten_default_variant=True)
            in TOGETHER_SUPPORTED_MODELS
        ), f"Unsupported model: {model_name}, use one of the supported models: {','.join(TOGETHER_SUPPORTED_MODELS.keys())}"

        return TOGETHER_SUPPORTED_MODELS.get(
            model.descriptor(shorten_default_variant=True)
        )

    def get_together_chat_options(self, request: ChatCompletionRequest) -> dict:
        options = {}
        if request.sampling_params is not None:
            for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
                if getattr(request.sampling_params, attr):
                    options[attr] = getattr(request.sampling_params, attr)

        return options

    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        # accumulate sampling params and other options to pass to together
        options = self.get_together_chat_options(request)
        together_model = self.resolve_together_model(request.model)

        if not request.stream:
            # TODO: might need to add back an async here
            r = self.client.chat.completions.create(
                model=together_model,
                messages=self._messages_to_together_messages(request.messages),
                stream=False,
                **options,
            )
            stop_reason = None
            if r.choices[0].finish_reason:
                if r.choices[0].finish_reason == "stop":
                    stop_reason = StopReason.end_of_turn
                elif r.choices[0].finish_reason == "length":
                    stop_reason = StopReason.out_of_tokens

            completion_message = decode_assistant_message_from_content(
                r.choices[0].message.content,
                stop_reason,
            )
            yield ChatCompletionResponse(
                completion_message=completion_message,
                logprobs=None,
            )
        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.start,
                    delta="",
                )
            )

            buffer = ""
            ipython = False
            stop_reason = None

            for chunk in self.client.chat.completions.create(
                model=together_model,
                messages=self._messages_to_together_messages(request.messages),
                stream=True,
                **options,
            ):
                if chunk.choices[0].finish_reason:
                    if stop_reason is None and chunk.choices[0].finish_reason == "stop":
                        stop_reason = StopReason.end_of_turn
                    elif (
                        stop_reason is None
                        and chunk.choices[0].finish_reason == "length"
                    ):
                        stop_reason = StopReason.out_of_tokens
                    break

                text = chunk.choices[0].delta.content
                if text is None:
                    continue

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

                if ipython:
                    if text == "<|eot_id|>":
                        stop_reason = StopReason.end_of_turn
                        text = ""
                        continue
                    elif text == "<|eom_id|>":
                        stop_reason = StopReason.end_of_message
                        text = ""
                        continue

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
            message = decode_assistant_message_from_content(buffer, stop_reason)
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


# TODO: Consolidate this with impl in llama-models
def decode_assistant_message_from_content(
    content: str,
    stop_reason: StopReason,
) -> CompletionMessage:
    ipython = content.startswith("<|python_tag|>")
    if ipython:
        content = content[len("<|python_tag|>") :]

    if content.endswith("<|eot_id|>"):
        content = content[: -len("<|eot_id|>")]
        stop_reason = StopReason.end_of_turn
    elif content.endswith("<|eom_id|>"):
        content = content[: -len("<|eom_id|>")]
        stop_reason = StopReason.end_of_message

    tool_name = None
    tool_arguments = {}

    custom_tool_info = ToolUtils.maybe_extract_custom_tool_call(content)
    if custom_tool_info is not None:
        tool_name, tool_arguments = custom_tool_info
        # Sometimes when agent has custom tools alongside builin tools
        # Agent responds for builtin tool calls in the format of the custom tools
        # This code tries to handle that case
        if tool_name in BuiltinTool.__members__:
            tool_name = BuiltinTool[tool_name]
            tool_arguments = {
                "query": list(tool_arguments.values())[0],
            }
    else:
        builtin_tool_info = ToolUtils.maybe_extract_builtin_tool_call(content)
        if builtin_tool_info is not None:
            tool_name, query = builtin_tool_info
            tool_arguments = {
                "query": query,
            }
            if tool_name in BuiltinTool.__members__:
                tool_name = BuiltinTool[tool_name]
        elif ipython:
            tool_name = BuiltinTool.code_interpreter
            tool_arguments = {
                "code": content,
            }

    tool_calls = []
    if tool_name is not None and tool_arguments is not None:
        call_id = str(uuid.uuid4())
        tool_calls.append(
            ToolCall(
                call_id=call_id,
                tool_name=tool_name,
                arguments=tool_arguments,
            )
        )
        content = ""

    if stop_reason is None:
        stop_reason = StopReason.out_of_tokens

    return CompletionMessage(
        content=content,
        stop_reason=stop_reason,
        tool_calls=tool_calls,
    )
