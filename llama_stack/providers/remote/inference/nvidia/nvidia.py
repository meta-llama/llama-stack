# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import warnings
from functools import lru_cache
from typing import AsyncIterator, List, Optional, Union

from openai import APIConnectionError, AsyncOpenAI, BadRequestError

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    TextTruncation,
    ToolChoice,
    ToolConfig,
)
from llama_stack.models.llama.datatypes import (
    SamplingParams,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
)
from llama_stack.providers.utils.inference.prompt_adapter import content_has_media

from . import NVIDIAConfig
from .models import MODEL_ENTRIES
from .openai_utils import (
    convert_chat_completion_request,
    convert_completion_request,
    convert_openai_completion_choice,
    convert_openai_completion_stream,
)
from .utils import _is_nvidia_hosted, check_health

logger = logging.getLogger(__name__)


class NVIDIAInferenceAdapter(Inference, ModelRegistryHelper):
    def __init__(self, config: NVIDIAConfig) -> None:
        # TODO(mf): filter by available models
        ModelRegistryHelper.__init__(self, model_entries=MODEL_ENTRIES)

        logger.info(f"Initializing NVIDIAInferenceAdapter({config.url})...")

        if _is_nvidia_hosted(config):
            if not config.api_key:
                raise RuntimeError(
                    "API key is required for hosted NVIDIA NIM. Either provide an API key or use a self-hosted NIM."
                )
        # elif self._config.api_key:
        #
        # we don't raise this warning because a user may have deployed their
        # self-hosted NIM with an API key requirement.
        #
        #     warnings.warn(
        #         "API key is not required for self-hosted NVIDIA NIM. "
        #         "Consider removing the api_key from the configuration."
        #     )

        self._config = config

    @lru_cache  # noqa: B019
    def _get_client(self, provider_model_id: str) -> AsyncOpenAI:
        """
        For hosted models, https://integrate.api.nvidia.com/v1 is the primary base_url. However,
        some models are hosted on different URLs. This function returns the appropriate client
        for the given provider_model_id.

        This relies on lru_cache and self._default_client to avoid creating a new client for each request
        or for each model that is hosted on https://integrate.api.nvidia.com/v1.

        :param provider_model_id: The provider model ID
        :return: An OpenAI client
        """

        @lru_cache  # noqa: B019
        def _get_client_for_base_url(base_url: str) -> AsyncOpenAI:
            """
            Maintain a single OpenAI client per base_url.
            """
            return AsyncOpenAI(
                base_url=base_url,
                api_key=(self._config.api_key.get_secret_value() if self._config.api_key else "NO KEY"),
                timeout=self._config.timeout,
            )

        special_model_urls = {
            "meta/llama-3.2-11b-vision-instruct": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
            "meta/llama-3.2-90b-vision-instruct": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct",
        }

        base_url = f"{self._config.url}/v1"
        if _is_nvidia_hosted(self._config) and provider_model_id in special_model_urls:
            base_url = special_model_urls[provider_model_id]

        return _get_client_for_base_url(base_url)

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if content_has_media(content):
            raise NotImplementedError("Media is not supported")

        await check_health(self._config)  # this raises errors

        provider_model_id = self.get_provider_model_id(model_id)
        request = convert_completion_request(
            request=CompletionRequest(
                model=provider_model_id,
                content=content,
                sampling_params=sampling_params,
                response_format=response_format,
                stream=stream,
                logprobs=logprobs,
            ),
            n=1,
        )

        try:
            response = await self._get_client(provider_model_id).completions.create(**request)
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to NVIDIA NIM at {self._config.url}: {e}") from e

        if stream:
            return convert_openai_completion_stream(response)
        else:
            # we pass n=1 to get only one completion
            return convert_openai_completion_choice(response.choices[0])

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        if any(content_has_media(content) for content in contents):
            raise NotImplementedError("Media is not supported")

        #
        # Llama Stack: contents = List[str] | List[InterleavedContentItem]
        #  ->
        # OpenAI: input = str | List[str]
        #
        # we can ignore str and always pass List[str] to OpenAI
        #
        flat_contents = [content.text if isinstance(content, TextContentItem) else content for content in contents]
        input = [content.text if isinstance(content, TextContentItem) else content for content in flat_contents]
        model = self.get_provider_model_id(model_id)

        extra_body = {}

        if text_truncation is not None:
            text_truncation_options = {
                TextTruncation.none: "NONE",
                TextTruncation.end: "END",
                TextTruncation.start: "START",
            }
            extra_body["truncate"] = text_truncation_options[text_truncation]

        if output_dimension is not None:
            extra_body["dimensions"] = output_dimension

        if task_type is not None:
            task_type_options = {
                EmbeddingTaskType.document: "passage",
                EmbeddingTaskType.query: "query",
            }
            extra_body["input_type"] = task_type_options[task_type]

        try:
            response = await self._client.embeddings.create(
                model=model,
                input=input,
                extra_body=extra_body,
            )
        except BadRequestError as e:
            raise ValueError(f"Failed to get embeddings: {e}") from e

        #
        # OpenAI: CreateEmbeddingResponse(data=[Embedding(embedding=List[float], ...)], ...)
        #  ->
        # Llama Stack: EmbeddingsResponse(embeddings=List[List[float]])
        #
        return EmbeddingsResponse(embeddings=[embedding.embedding for embedding in response.data])

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
        if sampling_params is None:
            sampling_params = SamplingParams()
        if tool_prompt_format:
            warnings.warn("tool_prompt_format is not supported by NVIDIA NIM, ignoring", stacklevel=2)

        await check_health(self._config)  # this raises errors

        provider_model_id = self.get_provider_model_id(model_id)
        request = await convert_chat_completion_request(
            request=ChatCompletionRequest(
                model=self.get_provider_model_id(model_id),
                messages=messages,
                sampling_params=sampling_params,
                response_format=response_format,
                tools=tools,
                stream=stream,
                logprobs=logprobs,
                tool_config=tool_config,
            ),
            n=1,
        )

        try:
            response = await self._get_client(provider_model_id).chat.completions.create(**request)
        except APIConnectionError as e:
            raise ConnectionError(f"Failed to connect to NVIDIA NIM at {self._config.url}: {e}") from e

        if stream:
            return convert_openai_chat_completion_stream(response, enable_incremental_tool_calls=False)
        else:
            # we pass n=1 to get only one completion
            return convert_openai_chat_completion_choice(response.choices[0])
