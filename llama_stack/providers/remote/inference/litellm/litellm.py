# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
from typing import AsyncIterator, List, Optional, Union, Any

from litellm import completion as litellm_completion
from litellm.types.utils import ModelResponse

from llama_models.datatypes import SamplingParams
from llama_models.llama3.api.datatypes import ToolDefinition, ToolPromptFormat, StopReason
from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionMessage,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    Inference,
    InterleavedContent,
    LogProbConfig,
    Message,
    ResponseFormat,
    ToolChoice,
)
# from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.remote.inference.litellm.config import LitellmConfig
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)

_MODEL_ALIASES = [
    build_model_alias(
        "gpt-4o",  # provider_model_id
        "gpt-4o",  # model_descriptor
    ),
]

class LitellmInferenceAdapter(Inference, ModelRegistryHelper):
    _config: LitellmConfig

    def __init__(self, config: LitellmConfig):
        ModelRegistryHelper.__init__(self, model_aliases=_MODEL_ALIASES)
        self._config = config

    def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        # litellm doesn't support non-chat completion as of time of writing
        raise NotImplementedError()

    def _messages_to_litellm_messages(
        self,
        messages: List[Message],
    ) -> list[dict[str, Any]]:
        litellm_messages = []
        for message in messages:
            lm_message = {
                "role": message.role,
                "content": message.content,
            }
            litellm_messages.append(lm_message)
        return litellm_messages

    def _convert_to_llama_stack_response(
        self,
        litellm_response: ModelResponse,
    ) -> ChatCompletionResponse:
        assert litellm_response.choices is not None
        assert len(litellm_response.choices) == 1
        message = litellm_response.choices[0].message
        completion_message = CompletionMessage(content=message["content"], role=message["role"], stop_reason=StopReason.end_of_message, tool_calls=[])
        return ChatCompletionResponse(completion_message=completion_message)

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:
        assert stream is False, "streaming not supported"
        model_id = self.get_provider_model_id(model_id)
        response = litellm_completion(
            model=model_id,
            custom_llm_provider=self._config.llm_provider,
            messages=self._messages_to_litellm_messages(messages),
            api_key=self._config.openai_api_key,
        )

        return self._convert_to_llama_stack_response(response)

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
        