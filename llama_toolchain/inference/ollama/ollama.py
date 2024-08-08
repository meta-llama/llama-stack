# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import AsyncGenerator, Dict

import httpx

from llama_models.llama3_1.api.datatypes import (
    BuiltinTool,
    CompletionMessage,
    Message,
    StopReason,
    ToolCall,
)
from llama_models.llama3_1.api.tool_utils import ToolUtils
from llama_models.sku_list import resolve_model
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
from ollama import AsyncClient

from .config import OllamaImplConfig

# TODO: Eventually this will move to the llama cli model list command
# mapping of Model SKUs to ollama models
OLLAMA_SUPPORTED_SKUS = {
    "Meta-Llama3.1-8B-Instruct": "llama3.1:8b-instruct-fp16",
    "Meta-Llama3.1-70B-Instruct": "llama3.1:70b-instruct-fp16",
}


async def get_provider_impl(
    config: OllamaImplConfig, _deps: Dict[Api, ProviderSpec]
) -> Inference:
    assert isinstance(
        config, OllamaImplConfig
    ), f"Unexpected config type: {type(config)}"
    impl = OllamaInference(config)
    await impl.initialize()
    return impl


class OllamaInference(Inference):

    def __init__(self, config: OllamaImplConfig) -> None:
        self.config = config

    @property
    def client(self) -> AsyncClient:
        return AsyncClient(host=self.config.url)

    async def initialize(self) -> None:
        try:
            await self.client.ps()
        except httpx.ConnectError:
            raise RuntimeError("Ollama Server is not running, start it using `ollama serve` in a separate terminal")

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    def _messages_to_ollama_messages(self, messages: list[Message]) -> list:
        ollama_messages = []
        for message in messages:
            if message.role == "ipython":
                role = "tool"
            else:
                role = message.role
            ollama_messages.append({"role": role, "content": message.content})

        return ollama_messages

    def resolve_ollama_model(self, model_name: str) -> str:
        model = resolve_model(model_name)
        assert (
            model is not None
            and model.descriptor(shorten_default_variant=True) in OLLAMA_SUPPORTED_SKUS
        ), f"Unsupported model: {model_name}, use one of the supported models: {','.join(OLLAMA_SUPPORTED_SKUS.keys())}"

        return OLLAMA_SUPPORTED_SKUS.get(model.descriptor(shorten_default_variant=True))

    def get_ollama_chat_options(self, request: ChatCompletionRequest) -> dict:
        options = {}
        if request.sampling_params is not None:
            for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
                if getattr(request.sampling_params, attr):
                    options[attr] = getattr(request.sampling_params, attr)
            if (
                request.sampling_params.repetition_penalty is not None
                and request.sampling_params.repetition_penalty != 1.0
            ):
                options["repeat_penalty"] = request.sampling_params.repetition_penalty

        return options

    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        # accumulate sampling params and other options to pass to ollama
        options = self.get_ollama_chat_options(request)
        ollama_model = self.resolve_ollama_model(request.model)

        res = await self.client.ps()
        need_model_pull = True
        for r in res["models"]:
            if ollama_model == r["model"]:
                need_model_pull = False
                break

        if need_model_pull:
            print(f"Pulling model: {ollama_model}")
            status = await self.client.pull(ollama_model)
            assert (
                status["status"] == "success"
            ), f"Failed to pull model {self.model} in ollama"

        if not request.stream:
            r = await self.client.chat(
                model=ollama_model,
                messages=self._messages_to_ollama_messages(request.messages),
                stream=False,
                options=options,
            )
            stop_reason = None
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
                model=ollama_model,
                messages=self._messages_to_ollama_messages(request.messages),
                stream=True,
                options=options,
            )

            buffer = ""
            ipython = False
            stop_reason = None

            async for chunk in stream:
                if chunk["done"]:
                    if stop_reason is None and chunk["done_reason"] == "stop":
                        stop_reason = StopReason.end_of_turn
                    elif stop_reason is None and chunk["done_reason"] == "length":
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
