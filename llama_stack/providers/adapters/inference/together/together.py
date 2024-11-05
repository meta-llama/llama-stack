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
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
)

from .config import TogetherImplConfig


TOGETHER_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "Llama3.1-70B-Instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "Llama3.1-405B-Instruct": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "Llama3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct-Turbo",
    "Llama3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
    "Llama3.2-90B-Vision-Instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
    "Llama-Guard-3-8B": "meta-llama/Meta-Llama-Guard-3-8B",
    "Llama-Guard-3-11B-Vision": "meta-llama/Llama-Guard-3-11B-Vision-Turbo",
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
        pass

    async def shutdown(self) -> None:
        pass

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        request = CompletionRequest(
            model=model,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    def _get_client(self) -> Together:
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
        return Together(api_key=together_api_key)

    async def _nonstream_completion(
        self, request: CompletionRequest
    ) -> ChatCompletionResponse:
        params = self._get_params_for_completion(request)
        r = self._get_client().completions.create(**params)
        return process_completion_response(r, self.formatter)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = self._get_params_for_completion(request)

        # if we shift to TogetherAsyncClient, we won't need this wrapper
        async def _to_async_generator():
            s = self._get_client().completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(stream, self.formatter):
            yield chunk

    def _build_options(
        self, sampling_params: Optional[SamplingParams], fmt: ResponseFormat
    ) -> dict:
        options = get_sampling_options(sampling_params)
        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                raise NotImplementedError("Grammar response format not supported yet")
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        return options

    def _get_params_for_completion(self, request: CompletionRequest) -> dict:
        return {
            "model": self.map_to_provider_model(request.model),
            "prompt": completion_request_to_prompt(request, self.formatter),
            "stream": request.stream,
            **self._build_options(request.sampling_params, request.response_format),
        }

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        response_format: Optional[ResponseFormat] = None,
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
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )

        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        params = self._get_params(request)
        r = self._get_client().completions.create(**params)
        return process_chat_completion_response(r, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        params = self._get_params(request)

        # if we shift to TogetherAsyncClient, we won't need this wrapper
        async def _to_async_generator():
            s = self._get_client().completions.create(**params)
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
            **self._build_options(request.sampling_params, request.response_format),
        }

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
