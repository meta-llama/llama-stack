# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

from llama_models.datatypes import CoreModelId

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from together import Together

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
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
    convert_message_to_dict,
    request_has_media,
)

from .config import TogetherImplConfig


MODEL_ALIASES = [
    build_model_alias(
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_model_alias(
        "meta-llama/Meta-Llama-Guard-3-8B",
        CoreModelId.llama_guard_3_8b.value,
    ),
    build_model_alias(
        "meta-llama/Llama-Guard-3-11B-Vision-Turbo",
        CoreModelId.llama_guard_3_11b_vision.value,
    ),
]


class TogetherInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):
    def __init__(self, config: TogetherImplConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ALIASES)
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def completion(
        self,
        model_id: str,
        content: InterleavedTextMedia,
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
        params = await self._get_params(request)
        r = self._get_client().completions.create(**params)
        return process_completion_response(r, self.formatter)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

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
            r = self._get_client().chat.completions.create(**params)
        else:
            r = self._get_client().completions.create(**params)
        return process_chat_completion_response(r, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
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
            if media_present:
                input_dict["messages"] = [
                    await convert_message_to_dict(m) for m in request.messages
                ]
            else:
                input_dict["prompt"] = chat_completion_request_to_prompt(
                    request, self.get_llama_model(request.model), self.formatter
                )
        else:
            assert (
                not media_present
            ), "Together does not support media for Completion requests"
            input_dict["prompt"] = completion_request_to_prompt(request, self.formatter)

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **self._build_options(request.sampling_params, request.response_format),
        }

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
