# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import AsyncGenerator

from fireworks.client import Fireworks

from llama_models.llama3.api.datatypes import (
    BuiltinTool,
    CompletionMessage,
    Message,
    StopReason,
    ToolCall,
)
from llama_models.llama3.api.tool_utils import ToolUtils
from llama_models.sku_list import resolve_model

from llama_toolchain.inference.api import *  # noqa: F403

from .config import FireworksImplConfig

FIREWORKS_SUPPORTED_MODELS = {
    "Meta-Llama3.1-8B-Instruct": "fireworks/llama-v3p1-8b-instruct",
    "Meta-Llama3.1-70B-Instruct": "fireworks/llama-v3p1-70b-instruct",
    "Meta-Llama3.1-405B-Instruct": "fireworks/llama-v3p1-405b-instruct",
}


class FireworksInferenceAdapter(Inference):
    def __init__(self, config: FireworksImplConfig) -> None:
        self.config = config

    @property
    def client(self) -> Fireworks:
        return Fireworks(api_key=self.config.api_key)

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    def _messages_to_fireworks_messages(self, messages: list[Message]) -> list:
        fireworks_messages = []
        for message in messages:
            if message.role == "ipython":
                role = "tool"
            else:
                role = message.role
            fireworks_messages.append({"role": role, "content": message.content})

        return fireworks_messages

    def resolve_fireworks_model(self, model_name: str) -> str:
        model = resolve_model(model_name)
        assert (
            model is not None
            and model.descriptor(shorten_default_variant=True)
            in FIREWORKS_SUPPORTED_MODELS
        ), f"Unsupported model: {model_name}, use one of the supported models: {','.join(FIREWORKS_SUPPORTED_MODELS.keys())}"

        return FIREWORKS_SUPPORTED_MODELS.get(
            model.descriptor(shorten_default_variant=True)
        )

    def get_fireworks_chat_options(self, request: ChatCompletionRequest) -> dict:
        options = {}
        if request.sampling_params is not None:
            for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
                if getattr(request.sampling_params, attr):
                    options[attr] = getattr(request.sampling_params, attr)

        return options

    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        # accumulate sampling params and other options to pass to fireworks
        options = self.get_fireworks_chat_options(request)
        fireworks_model = self.resolve_fireworks_model(request.model)

        if not request.stream:
            r = await self.client.chat.completions.acreate(
                model=fireworks_model,
                messages=self._messages_to_fireworks_messages(request.messages),
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

            async for chunk in self.client.chat.completions.acreate(
                model=fireworks_model,
                messages=self._messages_to_fireworks_messages(request.messages),
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
