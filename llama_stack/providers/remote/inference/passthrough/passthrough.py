# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, AsyncGenerator, Dict, List, Optional

from llama_stack_client import AsyncLlamaStackClient

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
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
from llama_stack.apis.models import Model
from llama_stack.distribution.library_client import convert_pydantic_to_json_value, convert_to_pydantic
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper

from .config import PassthroughImplConfig


class PassthroughInferenceAdapter(Inference):
    def __init__(self, config: PassthroughImplConfig) -> None:
        ModelRegistryHelper.__init__(self, [])
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        return model

    def _get_client(self) -> AsyncLlamaStackClient:
        passthrough_url = None
        passthrough_api_key = None
        provider_data = None

        if self.config.url is not None:
            passthrough_url = self.config.url
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.passthrough_url:
                raise ValueError(
                    'Pass url of the passthrough endpoint in the header X-LlamaStack-Provider-Data as { "passthrough_url": <your passthrough url>}'
                )
            passthrough_url = provider_data.passthrough_url

        if self.config.api_key is not None:
            passthrough_api_key = self.config.api_key.get_secret_value()
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.passthrough_api_key:
                raise ValueError(
                    'Pass API Key for the passthrough endpoint in the header X-LlamaStack-Provider-Data as { "passthrough_api_key": <your api key>}'
                )
            passthrough_api_key = provider_data.passthrough_api_key

        return AsyncLlamaStackClient(
            base_url=passthrough_url,
            api_key=passthrough_api_key,
            provider_data=provider_data,
        )

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        client = self._get_client()
        model = await self.model_store.get_model(model_id)

        request_params = {
            "model_id": model.provider_resource_id,
            "content": content,
            "sampling_params": sampling_params,
            "response_format": response_format,
            "stream": stream,
            "logprobs": logprobs,
        }

        request_params = {key: value for key, value in request_params.items() if value is not None}

        # cast everything to json dict
        json_params = self.cast_value_to_json_dict(request_params)

        # only pass through the not None params
        return await client.inference.completion(**json_params)

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)

        # TODO: revisit this remove tool_calls from messages logic
        for message in messages:
            if hasattr(message, "tool_calls"):
                message.tool_calls = None

        request_params = {
            "model_id": model.provider_resource_id,
            "messages": messages,
            "sampling_params": sampling_params,
            "tools": tools,
            "tool_choice": tool_choice,
            "tool_prompt_format": tool_prompt_format,
            "response_format": response_format,
            "stream": stream,
            "logprobs": logprobs,
        }

        # only pass through the not None params
        request_params = {key: value for key, value in request_params.items() if value is not None}

        # cast everything to json dict
        json_params = self.cast_value_to_json_dict(request_params)

        if stream:
            return self._stream_chat_completion(json_params)
        else:
            return await self._nonstream_chat_completion(json_params)

    async def _nonstream_chat_completion(self, json_params: Dict[str, Any]) -> ChatCompletionResponse:
        client = self._get_client()
        response = await client.inference.chat_completion(**json_params)

        return ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=response.completion_message.content.text,
                stop_reason=response.completion_message.stop_reason,
                tool_calls=response.completion_message.tool_calls,
            ),
            logprobs=response.logprobs,
        )

    async def _stream_chat_completion(self, json_params: Dict[str, Any]) -> AsyncGenerator:
        client = self._get_client()
        stream_response = await client.inference.chat_completion(**json_params)

        async for chunk in stream_response:
            chunk = chunk.to_dict()

            # temporary hack to remove the metrics from the response
            chunk["metrics"] = []
            chunk = convert_to_pydantic(ChatCompletionResponseStreamChunk, chunk)
            yield chunk

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedContent],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        client = self._get_client()
        model = await self.model_store.get_model(model_id)

        return await client.inference.embeddings(
            model_id=model.provider_resource_id,
            contents=contents,
            text_truncation=text_truncation,
            output_dimension=output_dimension,
            task_type=task_type,
        )

    def cast_value_to_json_dict(self, request_params: Dict[str, Any]) -> Dict[str, Any]:
        json_params = {}
        for key, value in request_params.items():
            json_input = convert_pydantic_to_json_value(value)
            if isinstance(json_input, dict):
                json_input = {k: v for k, v in json_input.items() if v is not None}
            elif isinstance(json_input, list):
                json_input = [x for x in json_input if x is not None]
                new_input = []
                for x in json_input:
                    if isinstance(x, dict):
                        x = {k: v for k, v in x.items() if v is not None}
                    new_input.append(x)
                json_input = new_input

            json_params[key] = json_input

        return json_params
