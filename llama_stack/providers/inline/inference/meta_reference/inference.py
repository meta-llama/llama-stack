# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

from llama_stack.apis.common.content_types import (
    TextDelta,
)
from llama_stack.apis.inference import (
    BatchChatCompletionResponse,
    BatchCompletionResponse,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    InferenceProvider,
    InterleavedContent,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    StopReason,
    TokenLogProbs,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
    UserMessage,
)
from llama_stack.apis.models import Model as ApiModel
from llama_stack.apis.models import ModelType
from llama_stack.log import get_logger
from llama_stack.models.llama.llama3.chat_format import ChatFormat as Llama3ChatFormat
from llama_stack.models.llama.llama3.tokenizer import Tokenizer as Llama3Tokenizer
from llama_stack.models.llama.llama4.chat_format import ChatFormat as Llama4ChatFormat
from llama_stack.models.llama.llama4.tokenizer import Tokenizer as Llama4Tokenizer
from llama_stack.models.llama.sku_list import resolve_model
from llama_stack.models.llama.sku_types import Model as LlamaModel
from llama_stack.models.llama.sku_types import ModelFamily
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.embedding_mixin import (
    SentenceTransformerEmbeddingMixin,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAIChatCompletionToLlamaStackMixin,
    OpenAICompletionToLlamaStackMixin,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    ChatCompletionRequestWithRawContent,
    CompletionRequestWithRawContent,
    augment_content_with_response_format_prompt,
    convert_request_to_raw,
)

from .config import MetaReferenceInferenceConfig
from .generators import LlamaGenerator
from .model_parallel import LlamaModelParallelGenerator

log = get_logger(__name__, category="inference")
# there's a single model parallel process running serving the model. for now,
# we don't support multiple concurrent requests to this process.
SEMAPHORE = asyncio.Semaphore(1)


def llama_builder_fn(config: MetaReferenceInferenceConfig, model_id: str, llama_model: LlamaModel) -> LlamaGenerator:
    return LlamaGenerator(config, model_id, llama_model)


class MetaReferenceInferenceImpl(
    OpenAICompletionToLlamaStackMixin,
    OpenAIChatCompletionToLlamaStackMixin,
    SentenceTransformerEmbeddingMixin,
    InferenceProvider,
    ModelsProtocolPrivate,
):
    def __init__(self, config: MetaReferenceInferenceConfig) -> None:
        self.config = config
        self.model_id: str | None = None
        self.llama_model: LlamaModel | None = None
        self.generator: LlamaGenerator | LlamaModelParallelGenerator | None = None
        self.model_registry_helper: ModelRegistryHelper | None = None

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        if self.config.create_distributed_process_group and self.generator:
            if hasattr(self.generator, "stop") and callable(self.generator.stop):
                self.generator.stop()

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def register_model(self, model: ApiModel) -> ApiModel:
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
            if model.provider_resource_id is not None:
                self._load_sentence_transformer_model(model.provider_resource_id)

        # TODO: what is this?! you can't really specify skipping via model metadata
        # kill this madness
        if "skip_load" in model.metadata and model.metadata["skip_load"]:
            return model

        await self.load_model(model.identifier, llama_model)
        return model

    async def load_model(self, model_id: str, llama_model: LlamaModel) -> None:
        log.info(f"Loading model `{model_id}`")

        builder_params: list[Any] = [self.config, model_id, llama_model]

        if self.config.create_distributed_process_group:
            self.generator = LlamaModelParallelGenerator(
                model_parallel_size=self.config.model_parallel_size or llama_model.pth_file_count,
                builder_fn=llama_builder_fn,
                builder_params=builder_params,
                formatter=(
                    Llama4ChatFormat(Llama4Tokenizer.get_instance())
                    if llama_model.model_family == ModelFamily.llama4
                    else Llama3ChatFormat(Llama3Tokenizer.get_instance())
                ),
            )
            self.generator.start()
        else:
            self.generator = llama_builder_fn(*builder_params)

        self.model_id = model_id
        self.llama_model = llama_model

        log.info("Warming up...")
        await self.completion(
            model_id=model_id,
            content="Hello, world!",
            sampling_params=SamplingParams(max_tokens=10),
        )
        await self.chat_completion(
            model_id=model_id,
            messages=[UserMessage(content="Hi how are you?")],
            sampling_params=SamplingParams(max_tokens=20),
        )
        log.info("Warmed up!")

    def check_model(self, request: CompletionRequest | ChatCompletionRequest) -> None:
        if self.model_id is None or self.llama_model is None:
            raise RuntimeError(
                "No avaible model yet, please register your requested model or add your model in the resouces first"
            )
        elif request.model != self.model_id:
            raise RuntimeError(f"Model mismatch: request model: {request.model} != loaded model: {self.model_id}")

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
    ) -> CompletionResponse | AsyncIterator[CompletionResponseStreamChunk]:
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
        request_with_raw_union = await convert_request_to_raw(request)

        # Type cast to ensure we have the correct type
        from typing import cast

        request_with_raw = cast(CompletionRequestWithRawContent, request_with_raw_union)

        if request.stream:
            return self._stream_completion(request_with_raw)
        else:
            results = await self._nonstream_completion([request_with_raw])
            return results[0]

    async def batch_completion(
        self,
        model_id: str,
        content_batch: list[InterleavedContent],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> BatchCompletionResponse:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        content_batch = [
            augment_content_with_response_format_prompt(response_format, content) for content in content_batch
        ]

        request_batch = []
        for content in content_batch:
            request = CompletionRequest(
                model=model_id,
                content=content,
                sampling_params=sampling_params,
                response_format=response_format,
                stream=False,
                logprobs=logprobs,
            )
            self.check_model(request)
            request_with_raw_union = await convert_request_to_raw(request)

            # Type cast to ensure we have the correct type
            from typing import cast

            request_with_raw = cast(CompletionRequestWithRawContent, request_with_raw_union)
            request_batch.append(request_with_raw)

        results = await self._nonstream_completion(request_batch)
        return BatchCompletionResponse(batch=results)

    async def _stream_completion(
        self, request: CompletionRequestWithRawContent
    ) -> AsyncIterator[CompletionResponseStreamChunk]:
        if not self.generator:
            raise RuntimeError("Generator not initialized")
        tokenizer = self.generator.formatter.tokenizer

        stop_reason = None

        for token_results in self.generator.completion([request]):
            token_result = token_results[0]
            if token_result.token == tokenizer.eot_id:
                stop_reason = StopReason.end_of_turn
                text = ""
            elif token_result.token == tokenizer.eom_id:
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

    async def _nonstream_completion(
        self, request_batch: list[CompletionRequestWithRawContent]
    ) -> list[CompletionResponse]:
        async with SEMAPHORE:
            if not self.generator:
                raise RuntimeError("Generator not initialized")

            class ItemState(BaseModel):
                tokens: list[int] = []
                logprobs: list[TokenLogProbs] = []
                stop_reason: StopReason | None = None
                finished: bool = False

            def impl() -> list[CompletionResponse]:
                if not self.generator:
                    raise RuntimeError("Generator not initialized")

                item_states = [ItemState() for _ in request_batch]

                for token_results in self.generator.completion(request_batch):
                    for idx, token_result in enumerate(token_results):
                        item_state = item_states[idx]
                        if item_state.finished:
                            continue

                        if token_result.token == self.generator.formatter.tokenizer.eot_id:
                            item_state.stop_reason = StopReason.end_of_turn
                            item_state.finished = True
                        elif token_result.token == self.generator.formatter.tokenizer.eom_id:
                            item_state.stop_reason = StopReason.end_of_message
                            item_state.finished = True
                        else:
                            item_state.tokens.append(token_result.token)
                            if request_batch[idx].logprobs:
                                assert len(token_result.logprobs) == 1
                                item_state.logprobs.append(
                                    TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]})
                                )

                # generate final responses
                completions = []
                for idx, item_state in enumerate(item_states):
                    if not self.generator:
                        raise RuntimeError("Generator not initialized")
                    content = self.generator.formatter.tokenizer.decode(item_state.tokens)

                    completions.append(
                        CompletionResponse(
                            content=content,
                            stop_reason=item_state.stop_reason or StopReason.out_of_tokens,
                            logprobs=item_state.logprobs if request_batch[idx].logprobs else None,
                        )
                    )

                return completions

            return await asyncio.get_event_loop().run_in_executor(None, impl)

    async def chat_completion(
        self,
        model_id: str,
        messages: list[Message],
        sampling_params: SamplingParams | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        response_format: ResponseFormat | None = None,
        stream: bool | None = False,
        logprobs: LogProbConfig | None = None,
        tool_config: ToolConfig | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionResponseStreamChunk]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        if self.llama_model is None:
            raise RuntimeError("Model not initialized")

        request = ChatCompletionRequest(
            model=model_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_config=tool_config or ToolConfig(),
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        self.check_model(request)
        request_with_raw_union = await convert_request_to_raw(request)

        # Type cast to ensure we have the correct type
        from typing import cast

        request_with_raw = cast(ChatCompletionRequestWithRawContent, request_with_raw_union)

        if request.stream:
            return self._stream_chat_completion(request_with_raw)
        else:
            results = await self._nonstream_chat_completion([request_with_raw])
            return results[0]

    async def batch_chat_completion(
        self,
        model_id: str,
        messages_batch: list[list[Message]],
        sampling_params: SamplingParams | None = None,
        tools: list[ToolDefinition] | None = None,
        tool_config: ToolConfig | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> BatchChatCompletionResponse:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if logprobs:
            assert logprobs.top_k == 1, f"Unexpected top_k={logprobs.top_k}"

        if self.llama_model is None:
            raise RuntimeError("Model not initialized")

        request_batch = []
        for messages in messages_batch:
            request = ChatCompletionRequest(
                model=model_id,
                messages=messages,
                sampling_params=sampling_params,
                tools=tools or [],
                tool_config=tool_config or ToolConfig(),
                response_format=response_format,
                stream=False,
                logprobs=logprobs,
            )
            self.check_model(request)
            request_with_raw_union = await convert_request_to_raw(request)

            # Type cast to ensure we have the correct type
            from typing import cast

            request_with_raw = cast(ChatCompletionRequestWithRawContent, request_with_raw_union)
            request_batch.append(request_with_raw)

        results = await self._nonstream_chat_completion(request_batch)
        return BatchChatCompletionResponse(batch=results)

    async def _nonstream_chat_completion(
        self, request_batch: list[ChatCompletionRequestWithRawContent]
    ) -> list[ChatCompletionResponse]:
        async with SEMAPHORE:
            if not self.generator:
                raise RuntimeError("Generator not initialized")

            class ItemState(BaseModel):
                tokens: list[int] = []
                logprobs: list[TokenLogProbs] = []
                stop_reason: StopReason | None = None
                finished: bool = False

            def impl() -> list[ChatCompletionResponse]:
                if not self.generator:
                    raise RuntimeError("Generator not initialized")

                item_states = [ItemState() for _ in request_batch]

                for token_results in self.generator.chat_completion(request_batch):
                    for idx, token_result in enumerate(token_results):
                        item_state = item_states[idx]
                        if item_state.finished:
                            continue

                        if token_result.token == self.generator.formatter.tokenizer.eot_id:
                            item_state.stop_reason = StopReason.end_of_turn
                            item_state.finished = True
                        elif token_result.token == self.generator.formatter.tokenizer.eom_id:
                            item_state.stop_reason = StopReason.end_of_message
                            item_state.finished = True
                        else:
                            item_state.tokens.append(token_result.token)
                            if request_batch[idx].logprobs:
                                assert len(token_result.logprobs) == 1
                                item_state.logprobs.append(
                                    TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]})
                                )

                # generate final responses
                completions = []
                for idx, item_state in enumerate(item_states):
                    if not self.generator:
                        raise RuntimeError("Generator not initialized")
                    content = self.generator.formatter.tokenizer.decode(item_state.tokens)

                    completions.append(
                        ChatCompletionResponse(
                            completion_message=CompletionMessage(
                                content=content,
                                stop_reason=item_state.stop_reason or StopReason.out_of_tokens,
                                tool_calls=[],
                            ),
                            logprobs=item_state.logprobs if request_batch[idx].logprobs else None,
                        )
                    )

                return completions

            return await asyncio.get_event_loop().run_in_executor(None, impl)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequestWithRawContent
    ) -> AsyncIterator[ChatCompletionResponseStreamChunk]:
        if not self.generator:
            raise RuntimeError("Generator not initialized")
        tokenizer = self.generator.formatter.tokenizer

        stop_reason = None

        for token_results in self.generator.chat_completion([request]):
            token_result = token_results[0]
            if token_result.token == tokenizer.eot_id:
                stop_reason = StopReason.end_of_turn
                text = ""
            elif token_result.token == tokenizer.eom_id:
                stop_reason = StopReason.end_of_message
                text = ""
            else:
                text = token_result.text

            logprobs = None
            if stop_reason is None:
                if request.logprobs:
                    assert len(token_result.logprobs) == 1

                    logprobs = [TokenLogProbs(logprobs_by_token={token_result.text: token_result.logprobs[0]})]

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=TextDelta(text=text),
                    logprobs=logprobs if request.logprobs else None,
                    stop_reason=stop_reason,
                )
            )
