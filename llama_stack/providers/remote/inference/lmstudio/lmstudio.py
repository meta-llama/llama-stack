# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncIterator, List, Optional, Union

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionResponse,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.inference.inference import (
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
    ResponseFormatType,
)
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.remote.inference.lmstudio._client import LMStudioClient
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.prompt_adapter import (
    content_has_media,
)

from .models import MODEL_ENTRIES


class LMStudioInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, url: str) -> None:
        self.url = url
        self.register_helper = ModelRegistryHelper(MODEL_ENTRIES)

    @property
    def client(self) -> LMStudioClient:
        return LMStudioClient(url=self.url)

    async def initialize(self) -> None:
        pass

    async def register_model(self, model):
        is_model_present = await self.client.check_if_model_present_in_lmstudio(model.provider_model_id)
        if not is_model_present:
            raise ValueError(f"Model with provider_model_id {model.provider_model_id} not found in LM Studio")
        await self.register_helper.register_model(model)
        return model

    async def unregister_model(self, model_id):
        pass

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        assert all(not content_has_media(content) for content in contents), (
            "Media content not supported in embedding model"
        )
        model = await self.model_store.get_model(model_id)
        embedding_model = await self.client.get_embedding_model(model.provider_model_id)
        embeddings = await self.client.embed(embedding_model, contents)
        return EmbeddingsResponse(embeddings=embeddings)

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
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        model = await self.model_store.get_model(model_id)
        llm = await self.client.get_llm(model.provider_model_id)

        if response_format is not None and response_format.type != ResponseFormatType.json_schema.value:
            raise ValueError(f"Response format type {response_format.type} not supported for LM Studio")
        json_schema = response_format.json_schema if response_format else None

        return await self.client.llm_respond(
            llm=llm,
            messages=messages,
            sampling_params=sampling_params,
            json_schema=json_schema,
            stream=stream,
            tool_config=tool_config,
            tools=tools,
        )

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,  # Skip this for now
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        model = await self.model_store.get_model(model_id)
        llm = await self.client.get_llm(model.provider_model_id)
        if content_has_media(content):
            raise NotImplementedError("Media content not supported in LM Studio")

        if response_format is not None and response_format.type != ResponseFormatType.json_schema.value:
            raise ValueError(f"Response format type {response_format.type} not supported for LM Studio")
        json_schema = response_format.json_schema if response_format else None

        return await self.client.llm_completion(llm, content, sampling_params, json_schema, stream)
