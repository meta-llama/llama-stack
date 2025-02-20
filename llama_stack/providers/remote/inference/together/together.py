# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List, Optional, Union

from together import Together

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    EmbeddingsResponse,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.model_registry import (
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

from .config import TogetherImplConfig
from .models import MODEL_ALIASES


class TogetherInferenceAdapter(ModelRegistryHelper, Inference, NeedsRequestProviderData):
    def __init__(self, config: TogetherImplConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ALIASES)
        self.config = config

    async def initialize(self) -> None:
        pass

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
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    def _get_client(self) -> Together:
        together_api_key = None
        if self.config.api_key is not None:
            together_api_key = self.config.api_key.get_secret_value()
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.together_api_key:
                raise ValueError(
                    'Pass Together API Key in the header X-LlamaStack-Provider-Data as { "together_api_key": <your api key>}'
                )
            together_api_key = provider_data.together_api_key
        return Together(api_key=together_api_key)

    async def _nonstream_completion(self, request: CompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = self._get_client().completions.create(**params)
        return process_completion_response(r)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        # if we shift to TogetherAsyncClient, we won't need this wrapper
        async def _to_async_generator():
            s = self._get_client().completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    def _build_options(
        self,
        sampling_params: Optional[SamplingParams],
        logprobs: Optional[LogProbConfig],
        fmt: ResponseFormat,
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

        if logprobs and logprobs.top_k:
            if logprobs.top_k != 1:
                raise ValueError(
                    f"Unsupported value: Together only supports logprobs top_k=1. {logprobs.top_k} was provided",
                )
            options["logprobs"] = 1

        return options

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )

        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        if "messages" in params:
            r = self._get_client().chat.completions.create(**params)
        else:
            r = self._get_client().completions.create(**params)
        return process_chat_completion_response(r, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        # if we shift to TogetherAsyncClient, we won't need this wrapper
        async def _to_async_generator():
            if "messages" in params:
                s = self._get_client().chat.completions.create(**params)
            else:
                s = self._get_client().completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> dict:
        input_dict = {}
        media_present = request_has_media(request)
        if isinstance(request, ChatCompletionRequest):
            if media_present:
                input_dict["messages"] = [await convert_message_to_openai_dict(m) for m in request.messages]
            else:
                input_dict["prompt"] = await chat_completion_request_to_prompt(
                    request, self.get_llama_model(request.model)
                )
        else:
            assert not media_present, "Together does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **self._build_options(request.sampling_params, request.logprobs, request.response_format),
        }

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)
        assert all(not content_has_media(content) for content in contents), (
            "Together does not support media for embeddings"
        )
        r = self._get_client().embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
        )
        embeddings = [item.embedding for item in r.data]
        return EmbeddingsResponse(embeddings=embeddings)
