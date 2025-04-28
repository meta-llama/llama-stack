# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, AsyncIterator, Dict, List, Optional, Union

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
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
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

    async def batch_completion(
        self,
        model_id: str,
        content_batch: List[InterleavedContent],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ):
        raise NotImplementedError("Batch completion is not supported by LM Studio Provider")

    async def batch_chat_completion(
        self,
        model_id: str,
        messages_batch: List[List[Message]],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_config: Optional[ToolConfig] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ):
        raise NotImplementedError("Batch completion is not supported by LM Studio Provider")

    async def openai_chat_completion(
        self,
        model: str,
        messages: List[OpenAIMessageParam],
        frequency_penalty: Optional[float] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[OpenAIResponseFormatParam] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
    ) -> Union[OpenAIChatCompletion, AsyncIterator[OpenAIChatCompletionChunk]]:
        if self.model_store is None:
            raise ValueError("ModelStore is not initialized")
        model_obj = await self.model_store.get_model(model)
        params = {
            k: v
            for k, v in {
                "model": model_obj.provider_resource_id,
                "messages": messages,
                "frequency_penalty": frequency_penalty,
                "function_call": function_call,
                "functions": functions,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "max_completion_tokens": max_completion_tokens,
                "max_tokens": max_tokens,
                "n": n,
                "parallel_tool_calls": parallel_tool_calls,
                "presence_penalty": presence_penalty,
                "response_format": response_format,
                "seed": seed,
                "stop": stop,
                "stream": stream,
                "stream_options": stream_options,
                "temperature": temperature,
                "tool_choice": tool_choice,
                "tools": tools,
                "top_logprobs": top_logprobs,
                "top_p": top_p,
                "user": user,
            }.items()
            if v is not None
        }
        return await self.openai_client.chat.completions.create(**params)  # type: ignore

    async def openai_completion(
        self,
        model: str,
        prompt: Union[str, List[str], List[int], List[List[int]]],
        best_of: Optional[int] = None,
        echo: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        prompt_logprobs: Optional[int] = None,
    ) -> OpenAICompletion:
        if not isinstance(prompt, str):
            raise ValueError("LM Studio does not support non-string prompts for completion")
        if self.model_store is None:
            raise ValueError("ModelStore is not initialized")
        model_obj = await self.model_store.get_model(model)
        params = {
            k: v
            for k, v in {
                "model": model_obj.provider_resource_id,
                "prompt": prompt,
                "best_of": best_of,
                "echo": echo,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "max_tokens": max_tokens,
                "n": n,
                "presence_penalty": presence_penalty,
                "seed": seed,
                "stop": stop,
                "stream": stream,
                "stream_options": stream_options,
                "temperature": temperature,
                "top_p": top_p,
                "user": user,
            }.items()
            if v is not None
        }
        return await self.openai_client.completions.create(**params)  # type: ignore

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
