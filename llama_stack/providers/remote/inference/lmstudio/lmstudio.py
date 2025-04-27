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
    GrammarResponseFormat,
    Inference,
    JsonSchemaResponseFormat,
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

    async def batch_chat_completion(self, *args, **kwargs):
        raise NotImplementedError("Batch chat completion not supported by LM Studio Provider")

    async def batch_completion(self, *args, **kwargs):
        raise NotImplementedError("Batch completion not supported by LM Studio Provider")

    async def openai_chat_completion(self, *args, **kwargs):
        raise NotImplementedError("OpenAI chat completion not supported by LM Studio Provider")

    async def openai_completion(self, *args, **kwargs):
        raise NotImplementedError("OpenAI completion not supported by LM Studio Provider")

    async def initialize(self) -> None:
        pass

    async def register_model(self, model):
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
        if self.model_store is None:
            raise ValueError("ModelStore is not initialized")
        model = await self.model_store.get_model(model_id)
        embedding_model = await self.client.get_embedding_model(model.provider_model_id)
        string_contents = [item.text if hasattr(item, "text") else str(item) for item in contents]
        embeddings = await self.client.embed(embedding_model, string_contents)
        return EmbeddingsResponse(embeddings=embeddings)

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,  # Default value changed from ToolChoice.auto to None
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        response_format: Optional[
            Union[JsonSchemaResponseFormat, GrammarResponseFormat]
        ] = None,  # Moved and type changed
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        if self.model_store is None:
            raise ValueError("ModelStore is not initialized")
        model = await self.model_store.get_model(model_id)
        llm = await self.client.get_llm(model.provider_model_id)

        json_schema_format = response_format if isinstance(response_format, JsonSchemaResponseFormat) else None
        if response_format is not None and not isinstance(response_format, JsonSchemaResponseFormat):
            raise ValueError(
                f"Response format type {type(response_format).__name__} not supported for LM Studio Provider"
            )
        return await self.client.llm_respond(
            llm=llm,
            messages=messages,
            sampling_params=sampling_params,
            json_schema=json_schema_format,
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
        if self.model_store is None:
            raise ValueError("ModelStore is not initialized")
        model = await self.model_store.get_model(model_id)
        llm = await self.client.get_llm(model.provider_model_id)
        if content_has_media(content):
            raise NotImplementedError("Media content not supported in LM Studio Provider")

        if not isinstance(response_format, JsonSchemaResponseFormat):
            raise ValueError(
                f"Response format type {type(response_format).__name__} not supported for LM Studio Provider"
            )

        return await self.client.llm_completion(llm, content, sampling_params, response_format, stream)
