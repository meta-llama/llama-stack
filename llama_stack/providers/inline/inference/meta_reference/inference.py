# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import logging
from typing import AsyncGenerator, List, Optional, Union

from llama_stack.apis.common.content_types import (
    TextDelta,
    ToolCallDelta,
    ToolCallParseStatus,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    Inference,
    InterleavedContent,
    LogProbConfig,
    Message,
    ResponseFormat,
    TokenLogProbs,
    ToolChoice,
    ToolConfig,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.models.llama.datatypes import (
    SamplingParams,
    StopReason,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.embedding_mixin import (
    SentenceTransformerEmbeddingMixin,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    augment_content_with_response_format_prompt,
    chat_completion_request_to_messages,
    convert_request_to_raw,
)

from .config import MetaReferenceInferenceConfig
from .llama3.generation import Llama3
from .model_parallel import LlamaModelParallelGenerator

log = logging.getLogger(__name__)
# there's a single model parallel process running serving the model. for now,
# we don't support multiple concurrent requests to this process.
SEMAPHORE = asyncio.Semaphore(1)


class MetaReferenceInferenceImpl(
    SentenceTransformerEmbeddingMixin,
    Inference,
    ModelsProtocolPrivate,
):
    def __init__(self, config: MetaReferenceInferenceConfig) -> None:
        self.config = config
        self.model_id = None
        self.llama_model = None

    async def initialize(self) -> None:
        pass

    async def load_model(self, model_id, llama_model) -> None:
        log.info(f"Loading model `{model_id}`")
        if self.config.create_distributed_process_group:
            self.generator = LlamaModelParallelGenerator(self.config, model_id, llama_model)
            self.generator.start()
        else:
            self.generator = Llama3.build(self.config, model_id, llama_model)

        self.model_id = model_id
        self.llama_model = llama_model

    async def shutdown(self) -> None:
        if self.config.create_distributed_process_group:
            self.generator.stop()

    def check_model(self, request) -> None:
        if self.model_id is None or self.llama_model is None:
            raise RuntimeError(
                "No avaible model yet, please register your requested model or add your model in the resouces first"
            )
        elif request.model != self.model_id:
            raise RuntimeError(f"Model mismatch: request model: {request.model} != loaded model: {self.model_id}")

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        llama_model = (
            resolve_model(model.metadata["llama_model"])
            if "llama_model" in model.metadata
            else resolve_model(model.identifier)
        )
        if llama_model is None:
            raise ValueError(
                "Please make sure your llama_model in model metadata or model identifier is in Llama SKU list"
            )

        self.model_registry_helper = ModelRegistryHelper(
            [
                build_hf_repo_model_entry(
                    llama_model.descriptor(),
                    llama_model.core_model_id.value,
                )
            ],
        )
        model = await self.model_registry_helper.register_model(model)

        if model.model_type == ModelType.embedding:
            self._load_sentence_transformer_model(model.provider_resource_id)

        if "skip_load" in model.metadata and model.metadata["skip_load"]:
            return model
        await self.load_model(model.identifier, llama_model)
        return model

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        content = augment_content_with_response_format_prompt(response_format, content)
        request = CompletionRequest(
            model=model_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        self.check_model(request)
        request = await convert_request_to_raw(request)

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

                        logprobs = [TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]})]

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

    async def _nonstream_completion(self, request: CompletionRequest) -> CompletionResponse:
        def impl():
            tokens = []
            logprobs = []
            stop_reason = None

            for token_result in self.generator.completion(request):
                tokens.append(token_result.token)
                if token_result.text == "<|eot_id|>":
                    stop_reason = StopReason.end_of_turn
                elif token_result.text == "<|eom_id|>":
                    stop_reason = StopReason.end_of_message

                if request.logprobs:
                    assert len(token_result.logprobs) == 1

                    logprobs.append(TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]}))

            if stop_reason is None:
                stop_reason = StopReason.out_of_tokens

            content = self.generator.formatter.tokenizer.decode(tokens)
            if content.endswith("<|eot_id|>"):
                content = content[: -len("<|eot_id|>")]
            elif content.endswith("<|eom_id|>"):
                content = content[: -len("<|eom_id|>")]
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
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request = ChatCompletionRequest(
            model=model_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )
        self.check_model(request)

        # augment and rewrite messages depending on the model
        request.messages = chat_completion_request_to_messages(request, self.llama_model.core_model_id.value)
        # download media and convert to raw content so we can send it to the model
        request = await convert_request_to_raw(request)

        if self.config.create_distributed_process_group:
            if SEMAPHORE.locked():
                raise RuntimeError("Only one concurrent request is supported")

        if request.stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
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

                    logprobs.append(TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]}))

            if stop_reason is None:
                stop_reason = StopReason.out_of_tokens

            raw_message = self.generator.formatter.decode_assistant_message(tokens, stop_reason)
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    content=raw_message.content,
                    stop_reason=raw_message.stop_reason,
                    tool_calls=raw_message.tool_calls,
                ),
                logprobs=logprobs if request.logprobs else None,
            )

        if self.config.create_distributed_process_group:
            async with SEMAPHORE:
                return impl()
        else:
            return impl()

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        def impl():
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.start,
                    delta=TextDelta(text=""),
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
                                tool_call="",
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
                        tool_call=text,
                        parse_status=ToolCallParseStatus.in_progress,
                    )
                else:
                    delta = TextDelta(text=text)

                if stop_reason is None:
                    if request.logprobs:
                        assert len(token_result.logprobs) == 1

                        logprobs.append(TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]}))
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

            message = self.generator.formatter.decode_assistant_message(tokens, stop_reason)

            parsed_tool_calls = len(message.tool_calls) > 0
            if ipython and not parsed_tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call="",
                            parse_status=ToolCallParseStatus.failed,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            for tool_call in message.tool_calls:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call=tool_call,
                            parse_status=ToolCallParseStatus.succeeded,
                        ),
                        stop_reason=stop_reason,
                    )
                )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta=TextDelta(text=""),
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
