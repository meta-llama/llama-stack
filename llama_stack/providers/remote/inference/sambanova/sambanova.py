# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import AsyncGenerator

from llama_models.datatypes import CoreModelId, SamplingStrategy

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message, ImageMedia
from llama_models.llama3.api.tokenizer import Tokenizer

from openai import OpenAI

from llama_stack.apis.inference import *  # noqa: F403

from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    process_chat_completion_response,
    process_chat_completion_stream_response,
)

from llama_stack.providers.utils.inference.prompt_adapter import (
    convert_message_to_dict,
)

from .config import SambaNovaImplConfig

MODEL_ALIASES = [
    build_model_alias(
        "Meta-Llama-3.1-8B-Instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "Meta-Llama-3.1-70B-Instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "Meta-Llama-3.1-405B-Instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_alias(
        "Meta-Llama-3.2-1B-Instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_alias(
        "Meta-Llama-3.2-3B-Instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "Llama-3.2-11B-Vision-Instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "Llama-3.2-90B-Vision-Instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
]


class SambaNovaInferenceAdapter(ModelRegistryHelper, Inference):
    def __init__(self, config: SambaNovaImplConfig) -> None:
        ModelRegistryHelper.__init__(
            self,
            model_aliases=MODEL_ALIASES,
        )
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    def _get_client(self) -> OpenAI:
        return OpenAI(base_url=self.config.url, api_key=self.config.api_key)

    async def completion(
        self,
        model_id: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()
    
    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
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
            stream=stream,
            logprobs=logprobs,
        )
        request_sambanova = await self.convert_chat_completion_request(request)

        client = OpenAI(base_url=self.config.url, api_key=self.config.api_key)
        if stream:
            return self._stream_chat_completion(request_sambanova, client)
        else:
            return await self._nonstream_chat_completion(request_sambanova, client)
        
    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: OpenAI
    ) -> ChatCompletionResponse:
        r = client.chat.completions.create(**request)
        return process_chat_completion_response(r, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: OpenAI
    ) -> AsyncGenerator:
        async def _to_async_generator():
            s = client.chat.completions.create(**request)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk
        
    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
    
    async def convert_chat_completion_request(self, request: ChatCompletionRequest) -> dict:
        compatible_request = self.convert_sampling_params(request.sampling_params)
        compatible_request["model"] = request.model
        compatible_request["messages"] = await self.convert_to_sambanova_message(request.messages)
        compatible_request["stream"] = request.stream
        compatible_request["logprobs"] = False
        compatible_request["extra_headers"] = {
            b"User-Agent": b"llama-stack: sambanova-inference-adapter",
        }
        return compatible_request

    def convert_sampling_params(self, sampling_params: SamplingParams, legacy: bool = False) -> dict:
        params = {}

        if sampling_params:
            params["frequency_penalty"] = sampling_params.repetition_penalty

            if sampling_params.max_tokens:
                if legacy:
                    params["max_tokens"] = sampling_params.max_tokens
                else:
                    params["max_completion_tokens"] = sampling_params.max_tokens

            if sampling_params.strategy == SamplingStrategy.top_p:
                params["top_p"] = sampling_params.top_p
            elif sampling_params.strategy == "top_k":
                params["extra_body"]["top_k"] = sampling_params.top_k
            elif sampling_params.strategy == "greedy":
                params["temperature"] = sampling_params.temperature
        
        return params

    async def convert_to_sambanova_message(self, messages: List[Message]) -> List[dict]:
        conversation = []
        for message in messages:
            content = await convert_message_to_dict(message)
            
            # Need to override role
            if isinstance(message, UserMessage):
                content["role"] = "user"
            elif isinstance(message, CompletionMessage):
                content["role"] = "assistant"
                tools = []
                for tool_call in message.tool_calls:
                    tools.append({
                        "id": tool_call.call_id,
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                        "type": "function",
                    })
                content["tool_calls"] = tools
            elif isinstance(message, ToolResponseMessage):
                content["role"] = "tool"
                content["tool_call_id"] = message.call_id
            elif isinstance(message, SystemMessage):
                content["role"] = "system"

            conversation.append(content)

        return conversation
        
    