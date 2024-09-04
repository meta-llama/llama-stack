# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List

import httpx

from huggingface_hub import InferenceClient

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message, StopReason
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_toolchain.inference.api import *  # noqa: F403


SUPPORTED_MODELS = {
    "Meta-Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Meta-Llama3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
}


class TGIInferenceAdapter(Inference):
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(tokenizer)
        self.model = None

    async def initialize(self) -> None:
        hf_models = {v: k for k, v in SUPPORTED_MODELS.items()}

        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.url}/info")
            response.raise_for_status()
            info = response.json()
            if "model_id" not in info:
                raise RuntimeError("Missing model_id in model info")
            model_id = info["model_id"]
            if model_id not in hf_models:
                raise RuntimeError(
                    f"TGI is serving model: {model_id}, use one of the supported models: {','.join(hf_models.keys())}"
                )

            self.model = hf_models[model_id]

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    def _convert_messages(self, messages: List[Message]) -> List[Message]:
        ret = []
        for message in messages:
            if message.role == "ipython":
                role = "tool"
            else:
                role = message.role
            ret.append({"role": role, "content": message.content})
        return ret

    def get_chat_options(self, request: ChatCompletionRequest) -> dict:
        options = {}
        if request.sampling_params is not None:
            for attr in {"temperature", "top_p", "top_k", "max_tokens"}:
                if getattr(request.sampling_params, attr):
                    options[attr] = getattr(request.sampling_params, attr)

        return options

    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        if request.model != self.model:
            raise ValueError(
                f"Model mismatch, expected: {self.model}, got: {request.model}"
            )

        options = self.get_chat_options(request)

        client = InferenceClient(base_url=self.url)
        if not request.stream:
            r = client.chat.completions.create(
                model=SUPPORTED_MODELS[self.model],
                messages=self._convert_messages(request.messages),
                stream=False,
                **options,
            )
            stop_reason = None
            if r.choices[0].finish_reason:
                if r.choices[0].finish_reason == "stop":
                    stop_reason = StopReason.end_of_turn
                elif r.choices[0].finish_reason == "length":
                    stop_reason = StopReason.out_of_tokens

            completion_message = self.formatter.decode_assistant_message_from_content(
                r.choices[0].message.content, stop_reason
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

            response = client.chat.completions.create(
                model=SUPPORTED_MODELS[self.model],
                messages=self._convert_messages(request.messages),
                stream=True,
                **options,
            )
            for chunk in response:
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
