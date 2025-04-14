# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import AsyncGenerator, List, Optional, Union

from llama_stack.apis.inference import (
    CompletionResponse,
    Inference,
    InterleavedContent,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.providers.datatypes import Model, ModelsProtocolPrivate
from llama_stack.providers.utils.inference.embedding_mixin import (
    SentenceTransformerEmbeddingMixin,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAIChatCompletionUnsupportedMixin,
    OpenAICompletionUnsupportedMixin,
)

from .config import SentenceTransformersInferenceConfig

log = logging.getLogger(__name__)


class SentenceTransformersInferenceImpl(
    OpenAIChatCompletionUnsupportedMixin,
    OpenAICompletionUnsupportedMixin,
    SentenceTransformerEmbeddingMixin,
    Inference,
    ModelsProtocolPrivate,
):
    def __init__(self, config: SentenceTransformersInferenceConfig) -> None:
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def register_model(self, model: Model) -> Model:
        return model

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def completion(
        self,
        model_id: str,
        content: str,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncGenerator]:
        raise ValueError("Sentence transformers don't support completion")

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
        raise ValueError("Sentence transformers don't support chat completion")

    async def batch_completion(
        self,
        model_id: str,
        content_batch: List[InterleavedContent],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ):
        raise NotImplementedError("Batch completion is not supported for Sentence Transformers")

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
        raise NotImplementedError("Batch chat completion is not supported for Sentence Transformers")
