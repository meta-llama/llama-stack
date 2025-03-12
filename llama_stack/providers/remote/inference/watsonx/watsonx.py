# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Union, AsyncIterator

from llama_stack.apis.common.content_types import InterleavedContent, InterleavedContentItem
from llama_stack.apis.inference import Inference, Message, ToolChoice, ResponseFormat, LogProbConfig, ToolConfig, \
    ChatCompletionResponse, ChatCompletionResponseStreamChunk, EmbeddingsResponse, TextTruncation, EmbeddingTaskType
from llama_stack.models.llama.datatypes import SamplingParams, ToolDefinition, ToolPromptFormat
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper

from . import WatsonXConfig

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from .models import MODEL_ENTRIES



class WatsonXInferenceAdapter(Inference, ModelRegistryHelper):
    def __init__(self, config: WatsonXConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES)

        print(f"Initializing WatsonXInferenceAdapter({config.url})...")

        self._config = config
        self._credential = {
            "url": self._config.url,
            "apikey": self._config.api_key
        }

        self._project_id = self._config.project_id
        self.params = {
            GenParams.MAX_NEW_TOKENS: 4096,
            GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
        }

    async def completion(
            self,
            model_id: str,
            content: InterleavedContent,
            sampling_params: Optional[SamplingParams] = None,
            response_format: Optional[ResponseFormat] = None,
            stream: Optional[bool] = False,
            logprobs: Optional[LogProbConfig] = None,
    ):
        pass

    async def embeddings(
            self,
            model_id: str,
            contents: List[str] | List[InterleavedContentItem],
            text_truncation: Optional[TextTruncation] = TextTruncation.none,
            output_dimension: Optional[int] = None,
            task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        pass

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
    ):
        # Language model
        model = Model(
            model_id=model_id,
            credentials=self._credential,
            project_id=self._project_id,
        )
        prompt = "\n".join(messages) + "\nAI: "

        response = model.generate_text(prompt=prompt, params=self.params)

        return response

