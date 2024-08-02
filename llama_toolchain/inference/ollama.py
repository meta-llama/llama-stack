# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

from typing import AsyncGenerator

import httpx

from llama_models.llama3_1.api.datatypes import (
    BuiltinTool,
    CompletionMessage,
    Message,
    StopReason,
    ToolCall,
)
from llama_models.llama3_1.api.tool_utils import ToolUtils

from ollama import AsyncClient

from .api.config import OllamaImplConfig
from .api.datatypes import (
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ToolCallDelta,
    ToolCallParseStatus,
)
from .api.endpoints import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    Inference,
)


def get_adapter_impl(config: OllamaImplConfig) -> Inference:
    assert isinstance(
        config, OllamaImplConfig
    ), f"Unexpected config type: {type(config)}"
    return OllamaInference(config)


class OllamaInference(Inference):

    def __init__(self, config: OllamaImplConfig) -> None:
        self.config = config
        self.model = config.model

    async def initialize(self) -> None:
        self.client = AsyncClient(host=self.config.url)
        try:
            status = await self.client.pull(self.model)
            assert (
                status["status"] == "success"
            ), f"Failed to pull model {self.model} in ollama"
        except httpx.ConnectError:
            print(
                "Ollama Server is not running, start it using `ollama serve` in a separate terminal"
            )
            raise

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    def _messages_to_ollama_messages(self, messages: list[Message]) -> list:
        ollama_messages = []
        for message in messages:
            ollama_messages.append({"role": message.role, "content": message.content})

        return ollama_messages

    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        if not request.stream:
            r = await self.client.chat(
                model=self.model,
                messages=self._messages_to_ollama_messages(request.messages),
                stream=False,
                # TODO: add support for options like temp, top_p, max_seq_length, etc
            )
            if r["done"]:
                if r["done_reason"] == "stop":
                    stop_reason = StopReason.end_of_turn
                elif r["done_reason"] == "length":
                    stop_reason = StopReason.out_of_tokens

            completion_message = decode_assistant_message_from_content(
                r["message"]["content"],
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

            stream = await self.client.chat(
                model=self.model,
                messages=self._messages_to_ollama_messages(request.messages),
                stream=True,
            )

            buffer = ""
            ipython = False
            stop_reason = None

            async for chunk in stream:
                # check if ollama is done
                if chunk["done"]:
                    if chunk["done_reason"] == "stop":
                        stop_reason = StopReason.end_of_turn
                    elif chunk["done_reason"] == "length":
                        stop_reason = StopReason.out_of_tokens
                    break

                text = chunk["message"]["content"]

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
                    buffer = buffer[len("<|python_tag|>") :]
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
