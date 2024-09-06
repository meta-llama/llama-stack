# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import AsyncGenerator

from huggingface_hub import InferenceClient
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import StopReason
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_toolchain.inference.api import *
from llama_toolchain.inference.api.api import (  # noqa: F403
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
)
from llama_toolchain.inference.prepare_messages import prepare_messages

from .config import TGIImplConfig

HF_SUPPORTED_MODELS = {
    "Meta-Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Meta-Llama3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct",
}


class TGIAdapter(Inference):

    def __init__(self, config: TGIImplConfig) -> None:
        self.config = config
        self.tokenizer = Tokenizer.get_instance()
        self.formatter = ChatFormat(self.tokenizer)

    @property
    def client(self) -> InferenceClient:
        return InferenceClient(base_url=self.config.url, token=self.config.api_token)

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

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

        model_info = self.client.get_endpoint_info(model=self.config.url)
        max_new_tokens = min(
            request.sampling_params.max_tokens or model_info["max_total_tokens"],
            model_info["max_total_tokens"] - len(model_input.tokens) - 1,
        )

        options = self.get_chat_options(request)

        if not request.stream:
            response = self.client.text_generation(
                prompt=prompt,
                stream=False,
                details=True,
                max_new_tokens=max_new_tokens,
                stop_sequences=["<|eom_id|>", "<|eot_id|>"],
                **options,
            )
            stop_reason = None
            if response.details.finish_reason:
                if response.details.finish_reason == "stop":
                    stop_reason = StopReason.end_of_turn
                elif response.details.finish_reason == "length":
                    stop_reason = StopReason.out_of_tokens

            completion_message = self.formatter.decode_assistant_message_from_content(
                response.generated_text,
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
            tokens = []

            for response in self.client.text_generation(
                prompt=prompt,
                stream=True,
                details=True,
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
