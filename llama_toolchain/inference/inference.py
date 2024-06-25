# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

from llama_models.llama3_1.api.datatypes import StopReason

from .api.config import InlineImplConfig
from .api.datatypes import (
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ToolCallDelta,
    ToolCallParseStatus,
)
from .api.endpoints import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    Inference,
)
from .model_parallel import LlamaModelParallelGenerator


class InferenceImpl(Inference):

    def __init__(self, config: InlineImplConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        self.generator = LlamaModelParallelGenerator(self.config)
        self.generator.start()

    async def shutdown(self) -> None:
        self.generator.stop()

    async def completion(self, request: CompletionRequest) -> AsyncGenerator:
        raise NotImplementedError()

    async def chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.start,
                delta="",
            )
        )

        tokens = []
        logprobs = []

        stop_reason = None

        buffer = ""
        ipython = False

        for token_result in self.generator.chat_completion(
            messages=request.messages,
            temperature=request.sampling_params.temperature,
            top_p=request.sampling_params.top_p,
            max_gen_len=request.sampling_params.max_tokens,
            logprobs=request.logprobs,
        ):
            buffer += token_result.text
            tokens.append(token_result.token)

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

            if not request.stream:
                if request.logprobs:
                    logprobs.append(token_result.logprob)

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
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=delta,
                    stop_reason=stop_reason,
                )
            )

        if stop_reason is None:
            stop_reason = StopReason.out_of_tokens

        # TODO(ashwin): parse tool calls separately here and report errors?
        # if someone breaks the iteration before coming here we are toast
        message = self.generator.formatter.decode_assistant_message(tokens, stop_reason)
        if request.stream:
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

            # TODO(ashwin): what else do we need to send out here when everything finishes?
        else:
            yield ChatCompletionResponse(
                content=message.content,
                tool_calls=message.tool_calls,
                stop_reason=stop_reason,
                logprobs=logprobs if request.logprobs else None,
            )
