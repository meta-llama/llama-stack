# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

import httpx

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message, StopReason
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.sku_list import resolve_model

from ollama import AsyncClient

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.utils.inference.prepare_messages import prepare_messages

# TODO: Eventually this will move to the llama cli model list command
# mapping of Model SKUs to ollama models
OLLAMA_SUPPORTED_SKUS = {
    # "Meta-Llama3.1-8B-Instruct": "llama3.1",
    "Meta-Llama3.1-8B-Instruct": "llama3.1:8b-instruct-fp16",
    "Meta-Llama3.1-70B-Instruct": "llama3.1:70b-instruct-fp16",
}


class OllamaInferenceAdapter(Inference):
    def __init__(self, url: str) -> None:
        self.url = url
        tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(tokenizer)

    @property
    def client(self) -> AsyncClient:
        return AsyncClient(host=self.url)

    async def initialize(self) -> None:
        try:
            await self.client.ps()
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Ollama Server is not running, start it using `ollama serve` in a separate terminal"
            ) from e

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

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )

        messages = prepare_messages(request)
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
                messages=self._messages_to_ollama_messages(messages),
                stream=False,
                options=options,
            )
            stop_reason = None
            if r["done"]:
                if r["done_reason"] == "stop":
                    stop_reason = StopReason.end_of_turn
                elif r["done_reason"] == "length":
                    stop_reason = StopReason.out_of_tokens

            completion_message = self.formatter.decode_assistant_message_from_content(
                r["message"]["content"], stop_reason
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
                messages=self._messages_to_ollama_messages(messages),
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
            message = self.formatter.decode_assistant_message_from_content(
                buffer, stop_reason
            )
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
