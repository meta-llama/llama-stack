# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List, Optional, Union

from fireworks.client import Fireworks
from llama_models.datatypes import CoreModelId

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
    request_has_media,
)
from ..nvidia.openai_utils import _convert_tooldef_to_openai_tool, convert_openai_chat_completion_choice

from .config import FireworksImplConfig


MODEL_ALIASES = [
    build_model_alias(
        "fireworks/llama-v3p1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "fireworks/llama-v3p1-70b-instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "fireworks/llama-v3p1-405b-instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_alias(
        "fireworks/llama-v3p2-1b-instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_alias(
        "fireworks/llama-v3p2-3b-instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "fireworks/llama-v3p2-11b-vision-instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "fireworks/llama-v3p2-90b-vision-instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_model_alias(
        "fireworks/llama-v3p3-70b-instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    build_model_alias(
        "fireworks/llama-guard-3-8b",
        CoreModelId.llama_guard_3_8b.value,
    ),
    build_model_alias(
        "fireworks/llama-guard-3-11b-vision",
        CoreModelId.llama_guard_3_11b_vision.value,
    ),
]


class FireworksInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):
    def __init__(self, config: FireworksImplConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ALIASES)
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def _get_api_key(self) -> str:
        if self.config.api_key is not None:
            return self.config.api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.fireworks_api_key:
                raise ValueError(
                    'Pass Fireworks API Key in the header X-LlamaStack-ProviderData as { "fireworks_api_key": <your api key>}'
                )
            return provider_data.fireworks_api_key

    def _get_client(self) -> Fireworks:
        fireworks_api_key = self._get_api_key()
        return Fireworks(api_key=fireworks_api_key)

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
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    async def _nonstream_completion(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        params = await self._get_params(request)
        r = await self._get_client().completion.acreate(**params)
        return process_completion_response(r, self.formatter)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        # Wrapper for async generator similar
        async def _to_async_generator():
            stream = self._get_client().completion.create(**params)
            for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(stream, self.formatter):
            yield chunk

    def _build_options(
        self, sampling_params: Optional[SamplingParams], fmt: ResponseFormat
    ) -> dict:
        options = get_sampling_options(sampling_params)
        options.setdefault("max_tokens", 512)

        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                options["response_format"] = {
                    "type": "grammar",
                    "grammar": fmt.bnf,
                }
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        return options

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
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)
        if "messages" in params:
            print(params)
            r = await self._get_client().chat.completions.acreate(**params)
            return convert_openai_chat_completion_choice(r.choices[0])
        else:
            r = await self._get_client().completion.acreate(**params)
            return process_chat_completion_response(r, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _to_async_generator():
            if "messages" in params:
                stream = self._get_client().chat.completions.acreate(**params)
            else:
                stream = self._get_client().completion.acreate(**params)
            async for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk

    async def _get_params(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        input_dict = {}
        media_present = request_has_media(request)

        if isinstance(request, ChatCompletionRequest):
            input_dict["messages"] = [
                await convert_message_to_openai_dict(m) for m in request.messages
            ]
            # print(input_dict["messages"])
            if request.tool_choice == ToolChoice.required:
                input_dict["tool_choice"] = "any"
            
            if request.tools:
                input_dict["tools"] = [
                    _convert_tooldef_to_openai_tool(t) for t in request.tools
                ]
            # print(input_dict)
        else:
            assert (
                not media_present
            ), "Fireworks does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(
                request, self.formatter
            )

        # Fireworks always prepends with BOS
        if "prompt" in input_dict:
            if input_dict["prompt"].startswith("<|begin_of_text|>"):
                input_dict["prompt"] = input_dict["prompt"][len("<|begin_of_text|>") :]

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **self._build_options(request.sampling_params, request.response_format),
        }

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)

        kwargs = {}
        if model.metadata.get("embedding_dimensions"):
            kwargs["dimensions"] = model.metadata.get("embedding_dimensions")
        assert all(
            not content_has_media(content) for content in contents
        ), "Fireworks does not support media for embeddings"
        response = self._get_client().embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
            **kwargs,
        )

        embeddings = [data.embedding for data in response.data]
        return EmbeddingsResponse(embeddings=embeddings)
