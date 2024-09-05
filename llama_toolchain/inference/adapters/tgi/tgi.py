# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List

import httpx

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message, StopReason
from llama_models.llama3.api.tokenizer import Tokenizer

from text_generation import Client

from llama_toolchain.inference.api import *  # noqa: F403
from llama_toolchain.inference.prepare_messages import prepare_messages


SUPPORTED_MODELS = {
    "Meta-Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Meta-Llama3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
}


class TGIInferenceAdapter(Inference):
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(self.tokenizer)
        self.model = None
        self.max_tokens = None

    async def initialize(self) -> None:
        hf_models = {v: k for k, v in SUPPORTED_MODELS.items()}

        try:
            print(f"Connecting to TGI server at: {self.url}")
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.url}/info")
                response.raise_for_status()
                info = response.json()
                if "model_id" not in info:
                    raise RuntimeError("Missing model_id in model info")
                if "max_total_tokens" not in info:
                    raise RuntimeError("Missing max_total_tokens in model info")
                self.max_tokens = info["max_total_tokens"]

                model_id = info["model_id"]
                if model_id not in hf_models:
                    raise RuntimeError(
                        f"TGI is serving model: {model_id}, use one of the supported models: {','.join(hf_models.keys())}"
                    )

                self.model = hf_models[model_id]
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise RuntimeError("Could not connect to TGI server") from e

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
        messages = prepare_messages(request)

        model_input = self.formatter.encode_dialog_prompt(messages)
        prompt = self.tokenizer.decode(model_input.tokens)
        max_new_tokens = min(
            request.sampling_params.max_tokens or self.max_tokens,
            self.max_tokens - len(model_input.tokens) - 1,
        )

        if request.model != self.model:
            raise ValueError(
                f"Model mismatch, expected: {self.model}, got: {request.model}"
            )

        options = self.get_chat_options(request)

        client = Client(base_url=self.url)
        if not request.stream:
            r = client.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                stop_sequences=["<|eom_id|>", "<|eot_id|>"],
                **options,
            )

            if r.details.finish_reason:
                if r.details.finish_reason == "stop":
                    stop_reason = StopReason.end_of_turn
                elif r.details.finish_reason == "length":
                    stop_reason = StopReason.out_of_tokens
                else:
                    stop_reason = StopReason.end_of_message
            else:
                stop_reason = StopReason.out_of_tokens

            completion_message = self.formatter.decode_assistant_message_from_content(
                r.generated_text, stop_reason
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
            tokens = []

            for response in client.generate_stream(
                prompt,
                max_new_tokens=max_new_tokens,
                stop_sequences=["<|eom_id|>", "<|eot_id|>"],
                **options,
            ):
                token_result = response.token

                buffer += token_result.text
                tokens.append(token_result.id)

                if not ipython and buffer.startswith("<|python_tag|>"):
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

                if token_result.text == "<|eot_id|>":
                    stop_reason = StopReason.end_of_turn
                    text = ""
                elif token_result.text == "<|eom_id|>":
                    stop_reason = StopReason.end_of_message
                    text = ""
                else:
                    text = token_result.text

                if ipython:
                    delta = ToolCallDelta(
                        content=text,
                        parse_status=ToolCallParseStatus.in_progress,
                    )
                else:
                    delta = text

                if stop_reason is None:
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=delta,
                            stop_reason=stop_reason,
                        )
                    )

            if stop_reason is None:
                stop_reason = StopReason.out_of_tokens

            # parse tool calls and report errors
            message = self.formatter.decode_assistant_message(tokens, stop_reason)
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
