# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, AsyncIterator, List, Optional, Union

import litellm

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
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
from llama_stack.apis.models.models import Model
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.log import get_logger
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    convert_chat_completion_request_to_openai_params,
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)

logger = get_logger(name=__name__, category="inference")


class LiteLLMOpenAIMixin(
    ModelRegistryHelper,
    Inference,
    NeedsRequestProviderData,
):
    def __init__(self, model_entries, api_key_from_config: str, provider_data_api_key_field: str):
        ModelRegistryHelper.__init__(self, model_entries)
        self.api_key_from_config = api_key_from_config
        self.provider_data_api_key_field = provider_data_api_key_field

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def register_model(self, model: Model) -> Model:
        model_id = self.get_provider_model_id(model.provider_resource_id)
        if model_id is None:
            raise ValueError(f"Unsupported model: {model.provider_resource_id}")
        return model

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError("LiteLLM does not support completion requests")

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
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        params = await convert_chat_completion_request_to_openai_params(request)

        # add api_key to params if available
        provider_data = self.get_request_provider_data()
        key_field = self.provider_data_api_key_field
        if provider_data and getattr(provider_data, key_field, None):
            api_key = getattr(provider_data, key_field)
        else:
            api_key = self.api_key_from_config
        params["api_key"] = api_key

        logger.debug(f"params to litellm (openai compat): {params}")
        # unfortunately, we need to use synchronous litellm.completion here because litellm
        # caches various httpx.client objects in a non-eventloop aware manner
        response = litellm.completion(**params)
        if stream:
            return self._stream_chat_completion(response)
        else:
            return convert_openai_chat_completion_choice(response.choices[0])

    async def _stream_chat_completion(
        self, response: litellm.ModelResponse
    ) -> AsyncIterator[ChatCompletionResponseStreamChunk]:
        async def _stream_generator():
            for chunk in response:
                yield chunk

        async for chunk in convert_openai_chat_completion_stream(
            _stream_generator(), enable_incremental_tool_calls=True
        ):
            yield chunk

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)

        response = litellm.embedding(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
        )

        embeddings = [data["embedding"] for data in response["data"]]
        return EmbeddingsResponse(embeddings=embeddings)
