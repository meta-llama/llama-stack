# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

from fireworks.client import Fireworks

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.inference import *  # noqa: F403

from llama_stack.providers.utils.inference.augment_messages import (
    chat_completion_request_to_prompt,
)
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)

from .config import FireworksImplConfig


FIREWORKS_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "fireworks/llama-v3p1-8b-instruct",
    "Llama3.1-70B-Instruct": "fireworks/llama-v3p1-70b-instruct",
    "Llama3.1-405B-Instruct": "fireworks/llama-v3p1-405b-instruct",
    "Llama3.2-1B-Instruct": "fireworks/llama-v3p2-1b-instruct",
    "Llama3.2-3B-Instruct": "fireworks/llama-v3p2-3b-instruct",
}


class FireworksInferenceAdapter(ModelRegistryHelper, Inference):
    def __init__(self, config: FireworksImplConfig) -> None:
        ModelRegistryHelper.__init__(
            self, stack_to_provider_models_map=FIREWORKS_SUPPORTED_MODELS
        )
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()

    def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )

        client = Fireworks(api_key=self.config.api_key)
        if stream:
            return self._stream_chat_completion(request, client)
        else:
            return self._nonstream_chat_completion(request, client)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: Fireworks
    ) -> ChatCompletionResponse:
        params = self._get_params(request)
        r = await client.completion.acreate(**params)
        return process_chat_completion_response(request, r, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: Fireworks
    ) -> AsyncGenerator:
        params = self._get_params(request)

        stream = client.completion.acreate(**params)
        async for chunk in process_chat_completion_stream_response(
            request, stream, self.formatter
        ):
            yield chunk

    def _get_params(self, request: ChatCompletionRequest) -> dict:
        prompt = chat_completion_request_to_prompt(request, self.formatter)
        # Fireworks always prepends with BOS
        if prompt.startswith("<|begin_of_text|>"):
            prompt = prompt[len("<|begin_of_text|>") :]

        options = get_sampling_options(request)
        options.setdefault("max_tokens", 512)
        return {
            "model": self.map_to_provider_model(request.model),
            "prompt": prompt,
            "stream": request.stream,
            **options,
        }
