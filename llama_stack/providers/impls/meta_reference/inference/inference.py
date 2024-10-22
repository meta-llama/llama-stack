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

from .config import MetaReferenceInferenceConfig
from .generation import Llama
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
        if self.config.create_distributed_process_group:
            self.generator = LlamaModelParallelGenerator(self.config)
            self.generator.start()
        else:
            self.generator = Llama.build(self.config)

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
        if self.config.create_distributed_process_group:
            self.generator.stop()

    def check_model(self, request) -> None:
        model = resolve_model(request.model)
        if model is None:
            raise RuntimeError(
                f"Unknown model: {request.model}, Run `llama model list`"
            )
        elif model.descriptor() != self.model.descriptor():
            raise RuntimeError(
                f"Model mismatch: {request.model} != {self.model.descriptor()}"
            )

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        request = CompletionRequest(
            model=model,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        self.check_model(request)

        if request.stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        def impl():
            stop_reason = None

            for token_result in self.generator.completion(request):
                if token_result.text == "<|eot_id|>":
                    stop_reason = StopReason.end_of_turn
                    text = ""
                elif token_result.text == "<|eom_id|>":
                    stop_reason = StopReason.end_of_message
                    text = ""
                else:
                    text = token_result.text

                logprobs = None
                if stop_reason is None:
                    if request.logprobs:
                        assert len(token_result.logprobs) == 1

                        logprobs = [
                            TokenLogProbs(
                                logprobs_by_token={
                                    token_result.text: token_result.logprobs[0]
                                }
                            )
                        ]

                yield CompletionResponseStreamChunk(
                    delta=text,
                    stop_reason=stop_reason,
                    logprobs=logprobs if request.logprobs else None,
                )

            if stop_reason is None:
                yield CompletionResponseStreamChunk(
                    delta="",
                    stop_reason=StopReason.out_of_tokens,
                )

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                for x in impl():
                    yield x
        else:
            for x in impl():
                yield x

    async def _nonstream_completion(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        def impl():
            tokens = []
            logprobs = []
            stop_reason = None

            tokenizer = self.generator.formatter.tokenizer
            for token_result in self.generator.completion(request):
                tokens.append(token_result.token)

                if token_result.token in tokenizer.stop_tokens:
                    # not quite right semantically
                    stop_reason = StopReason.end_of_turn

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

            content = self.generator.formatter.tokenizer.decode(tokens)
            return CompletionResponse(
                content=content,
                stop_reason=stop_reason,
                logprobs=logprobs if request.logprobs else None,
            )

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                return impl()
        else:
            return impl()

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
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
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        self.check_model(request)

        if self.config.create_distributed_process_group:
            if SEMAPHORE.locked():
                raise RuntimeError("Only one concurrent request is supported")

        if request.stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        def impl():
            tokens = []
            logprobs = []
            stop_reason = None

            for token_result in self.generator.chat_completion(request):
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

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                return impl()
        else:
            return impl()

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        def impl():
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

            for token_result in self.generator.chat_completion(request):
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

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                for x in impl():
                    yield x
        else:
            for x in impl():
                yield x

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
