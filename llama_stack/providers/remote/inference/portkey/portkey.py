# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

from portkey_ai import AsyncPortkey

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.inference import *  # noqa: F403

from llama_models.datatypes import CoreModelId

from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)
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

from .config import PortkeyImplConfig


model_aliases = [
    build_model_alias(
        "llama3.1-8b",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "llama-3.3-70b",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    build_model_alias(
        "llama"
    )
]


class PortkeyInferenceAdapter(ModelRegistryHelper, Inference):
    def __init__(self, config: PortkeyImplConfig) -> None:
        ModelRegistryHelper.__init__(
            self,
            model_aliases=model_aliases,
        )
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

        self.client = AsyncPortkey(
            base_url=self.config.base_url, api_key=self.config.api_key
        )

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        model = await self.model_store.get_model(model_id)
        request = CompletionRequest(
            model=model.provider_resource_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(
                request,
            )
        else:
            return await self._nonstream_completion(request)

    async def _nonstream_completion(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        params = await self._get_params(request)

        r = await self.client.completions.create(**params)

        return process_completion_response(r, self.formatter)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        stream = await self.client.completions.create(**params)

        async for chunk in process_completion_stream_response(stream, self.formatter):
            yield chunk

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
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
        self, request: CompletionRequest
    ) -> CompletionResponse:
        params = await self._get_params(request)

        r = await self.client.completions.create(**params)

        return process_chat_completion_response(r, self.formatter)

    async def _stream_chat_completion(
        self, request: CompletionRequest
    ) -> AsyncGenerator:
        params = await self._get_params(request)

        stream = await self.client.completions.create(**params)

        async for chunk in process_chat_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk

    async def _get_params(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        if request.sampling_params and request.sampling_params.top_k:
            raise ValueError("`top_k` not supported by Portkey")

        prompt = ""
        if isinstance(request, ChatCompletionRequest):
            prompt = await chat_completion_request_to_prompt(
                request, self.get_llama_model(request.model), self.formatter
            )
        elif isinstance(request, CompletionRequest):
            prompt = await completion_request_to_prompt(request, self.formatter)
        else:
            raise ValueError(f"Unknown request type {type(request)}")

        return {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

    # async def embeddings(
    #     self,
    #     model_id: str,
    #     contents: List[InterleavedContent],
    # ) -> EmbeddingsResponse:
    #     raise NotImplementedError()
