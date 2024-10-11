# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from together import Together

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
)

from .config import TogetherImplConfig


TOGETHER_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Llama3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "Llama3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Llama3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "Llama3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "Llama3.2-90B-Vision-Instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
}


class TogetherInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):
    def __init__(self, config: TogetherImplConfig) -> None:
        ModelRegistryHelper.__init__(
            self, stack_to_provider_models_map=TOGETHER_SUPPORTED_MODELS
        )
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    async def completion(
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
        together_api_key = None
        if self.config.api_key is not None:
            together_api_key = self.config.api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.together_api_key:
                raise ValueError(
                    'Pass Together API Key in the header X-LlamaStack-ProviderData as { "together_api_key": <your api key>}'
                )
            together_api_key = provider_data.together_api_key

        client = Together(api_key=together_api_key)
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

        if stream:
            return self._stream_chat_completion(request, client)
        else:
            return self._nonstream_chat_completion(request, client)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: Together
    ) -> ChatCompletionResponse:
        params = self._get_params(request)
        r = client.completions.create(**params)
        return process_chat_completion_response(r, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: Together
    ) -> AsyncGenerator:
        params = self._get_params(request)

        # if we shift to TogetherAsyncClient, we won't need this wrapper
        async def _to_async_generator():
            s = client.completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk

    def _get_params(self, request: ChatCompletionRequest) -> dict:
        return {
            "model": self.map_to_provider_model(request.model),
            "prompt": chat_completion_request_to_prompt(request, self.formatter),
            "stream": request.stream,
            **get_sampling_options(request),
        }

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
