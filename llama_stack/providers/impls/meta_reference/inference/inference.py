# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from typing import AsyncGenerator, List

from llama_models.sku_list import resolve_model

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.datatypes import ModelDef, ModelsProtocolPrivate
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_messages,
)

from .config import MetaReferenceInferenceConfig
from .model_parallel import LlamaModelParallelGenerator

# there's a single model parallel process running serving the model. for now,
# we don't support multiple concurrent requests to this process.
SEMAPHORE = asyncio.Semaphore(1)


class MetaReferenceInferenceImpl(Inference, ModelsProtocolPrivate):
    def __init__(self, config: MetaReferenceInferenceConfig) -> None:
        self.config = config
        model = resolve_model(config.model)
        if model is None:
            raise RuntimeError(f"Unknown model: {config.model}, Run `llama model list`")
        self.model = model
        # verify that the checkpoint actually is for this model lol

    async def initialize(self) -> None:
        print(f"Loading model `{self.model.descriptor()}`")
        self.generator = LlamaModelParallelGenerator(self.config)
        self.generator.start()

    async def register_model(self, model: ModelDef) -> None:
        raise ValueError("Dynamic model registration is not supported")

    async def list_models(self) -> List[ModelDef]:
        return [
            ModelDef(
                identifier=self.model.descriptor(),
                llama_model=self.model.descriptor(),
            )
        ]

    async def shutdown(self) -> None:
        self.generator.stop()

    def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
        raise NotImplementedError()

    def chat_completion(
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
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        # wrapper request to make it easier to pass around (internal only, not exposed to API)
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

        model = resolve_model(request.model)
        if model is None:
            raise RuntimeError(
                f"Unknown model: {request.model}, Run `llama model list`"
            )
        elif model.descriptor() != self.model.descriptor():
            raise RuntimeError(
                f"Model mismatch: {request.model} != {self.model.descriptor()}"
            )

        if SEMAPHORE.locked():
            raise RuntimeError("Only one concurrent request is supported")

        if request.stream:
            return self._stream_chat_completion(request)
        else:
            return self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        async with SEMAPHORE:
            messages = chat_completion_request_to_messages(request)

            tokens = []
            logprobs = []
            stop_reason = None

            for token_result in self.generator.chat_completion(
                messages=messages,
                temperature=request.sampling_params.temperature,
                top_p=request.sampling_params.top_p,
                max_gen_len=request.sampling_params.max_tokens,
                logprobs=request.logprobs,
                tool_prompt_format=request.tool_prompt_format,
            ):
                tokens.append(token_result.token)

                if token_result.text == "<|eot_id|>":
                    stop_reason = StopReason.end_of_turn
                elif token_result.text == "<|eom_id|>":
                    stop_reason = StopReason.end_of_message

                if request.logprobs:
                    assert len(token_result.logprobs) == 1

                    logprobs.append(
                        TokenLogProbs(
                            logprobs_by_token={
                                token_result.text: token_result.logprobs[0]
                            }
                        )
                    )

            if stop_reason is None:
                stop_reason = StopReason.out_of_tokens

            message = self.generator.formatter.decode_assistant_message(
                tokens, stop_reason
            )
            return ChatCompletionResponse(
                completion_message=message,
                logprobs=logprobs if request.logprobs else None,
            )

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        async with SEMAPHORE:
            messages = chat_completion_request_to_messages(request)

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.start,
                    delta="",
                )
            )

            tokens = []
            logprobs = []
            stop_reason = None
            ipython = False

            for token_result in self.generator.chat_completion(
                messages=messages,
                temperature=request.sampling_params.temperature,
                top_p=request.sampling_params.top_p,
                max_gen_len=request.sampling_params.max_tokens,
                logprobs=request.logprobs,
                tool_prompt_format=request.tool_prompt_format,
            ):
                tokens.append(token_result.token)

                if not ipython and token_result.text.startswith("<|python_tag|>"):
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
                    if request.logprobs:
                        assert len(token_result.logprobs) == 1

                        logprobs.append(
                            TokenLogProbs(
                                logprobs_by_token={
                                    token_result.text: token_result.logprobs[0]
                                }
                            )
                        )
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=delta,
                            stop_reason=stop_reason,
                            logprobs=logprobs if request.logprobs else None,
                        )
                    )

            if stop_reason is None:
                stop_reason = StopReason.out_of_tokens

            message = self.generator.formatter.decode_assistant_message(
                tokens, stop_reason
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

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
