# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import AsyncGenerator

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from openai import OpenAI

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.datatypes import ModelsProtocolPrivate

from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
)

from .config import VLLMImplConfig

VLLM_SUPPORTED_MODELS = {
    "Llama3.1-8B": "meta-llama/Llama-3.1-8B",
    "Llama3.1-70B": "meta-llama/Llama-3.1-70B",
    "Llama3.1-405B:bf16-mp8": "meta-llama/Llama-3.1-405B",
    "Llama3.1-405B": "meta-llama/Llama-3.1-405B-FP8",
    "Llama3.1-405B:bf16-mp16": "meta-llama/Llama-3.1-405B",
    "Llama3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama3.1-70B-Instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "Llama3.1-405B-Instruct:bf16-mp8": "meta-llama/Llama-3.1-405B-Instruct",
    "Llama3.1-405B-Instruct": "meta-llama/Llama-3.1-405B-Instruct-FP8",
    "Llama3.1-405B-Instruct:bf16-mp16": "meta-llama/Llama-3.1-405B-Instruct",
    "Llama3.2-1B": "meta-llama/Llama-3.2-1B",
    "Llama3.2-3B": "meta-llama/Llama-3.2-3B",
    "Llama3.2-11B-Vision": "meta-llama/Llama-3.2-11B-Vision",
    "Llama3.2-90B-Vision": "meta-llama/Llama-3.2-90B-Vision",
    "Llama3.2-1B-Instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "Llama3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    "Llama3.2-11B-Vision-Instruct": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Llama3.2-90B-Vision-Instruct": "meta-llama/Llama-3.2-90B-Vision-Instruct",
    "Llama-Guard-3-11B-Vision": "meta-llama/Llama-Guard-3-11B-Vision",
    "Llama-Guard-3-1B:int4-mp1": "meta-llama/Llama-Guard-3-1B-INT4",
    "Llama-Guard-3-1B": "meta-llama/Llama-Guard-3-1B",
    "Llama-Guard-3-8B": "meta-llama/Llama-Guard-3-8B",
    "Llama-Guard-3-8B:int8-mp1": "meta-llama/Llama-Guard-3-8B-INT8",
    "Prompt-Guard-86M": "meta-llama/Prompt-Guard-86M",
    "Llama-Guard-2-8B": "meta-llama/Llama-Guard-2-8B",
}


class VLLMInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, config: VLLMImplConfig) -> None:
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())
        self.client = None

    async def initialize(self) -> None:
        self.client = OpenAI(base_url=self.config.url, api_key=self.config.api_token)

    async def register_model(self, model: ModelDef) -> None:
        raise ValueError("Model registration is not supported for vLLM models")

    async def shutdown(self) -> None:
        pass

    async def list_models(self) -> List[ModelDef]:
        return [
            ModelDef(identifier=model.id, llama_model=model.id)
            for model in self.client.models.list()
        ]

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
        raise NotImplementedError()

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
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
        if stream:
            return self._stream_chat_completion(request, self.client)
        else:
            return await self._nonstream_chat_completion(request, self.client)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: OpenAI
    ) -> ChatCompletionResponse:
        params = self._get_params(request)
        r = client.completions.create(**params)
        return process_chat_completion_response(request, r, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: OpenAI
    ) -> AsyncGenerator:
        params = self._get_params(request)

        # TODO: Can we use client.completions.acreate() or maybe there is another way to directly create an async
        #  generator so this wrapper is not necessary?
        async def _to_async_generator():
            s = client.completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(
            request, stream, self.formatter
        ):
            yield chunk

    def _get_params(self, request: ChatCompletionRequest) -> dict:
        return {
            "model": VLLM_SUPPORTED_MODELS[request.model],
            "prompt": chat_completion_request_to_prompt(request, self.formatter),
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
        }

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
