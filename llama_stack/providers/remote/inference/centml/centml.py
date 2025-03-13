# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List, Optional, Union

from openai import OpenAI
from pydantic import parse_obj_as

from llama_models.datatypes import CoreModelId
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
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
    build_model_entry,
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

from .config import CentMLImplConfig

# Update this if list of model changes.
MODEL_ALIASES = [
    build_model_entry(
        "meta-llama/Llama-3.2-3B-Instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_entry(
        "meta-llama/Llama-3.3-70B-Instruct",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
]


class CentMLInferenceAdapter(ModelRegistryHelper, Inference,
                             NeedsRequestProviderData):
    """
    Adapter to use CentML's serverless inference endpoints,
    which adhere to the OpenAI chat/completions API spec,
    inside llama-stack.
    """

    def __init__(self, config: CentMLImplConfig) -> None:
        super().__init__(MODEL_ALIASES)
        self.config = config

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def _get_api_key(self) -> str:
        """
        Obtain the CentML API key either from the adapter config
        or from the dynamic provider data in request headers.
        """
        if self.config.api_key is not None:
            return self.config.api_key.get_secret_value()
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.centml_api_key:
                raise ValueError(
                    'Pass CentML API Key in the header X-LlamaStack-ProviderData as { "centml_api_key": "<your-api-key>" }'
                )
            return provider_data.centml_api_key

    def _get_client(self) -> OpenAI:
        """
        Creates an OpenAI-compatible client pointing to CentML's base URL,
        using the user's CentML API key.
        """
        api_key = self._get_api_key()
        return OpenAI(api_key=api_key, base_url=self.config.url)

    #
    # COMPLETION (non-chat)
    #

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        """
        For "completion" style requests (non-chat).
        """
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
            self, request: CompletionRequest) -> CompletionResponse:
        """
        Process non-streaming completion requests.

        If a structured output is specified (e.g. JSON schema),
        the adapter calls the chat completions endpoint and then
        converts the chat response into a plain CompletionResponse.
        Otherwise, it uses the regular completions endpoint.
        """
        params = await self._get_params(request)
        if request.response_format is not None:
            # ***** HACK: Use the chat completions endpoint even for non-chat completions
            # This is necessary because CentML's structured output (JSON schema) support
            # is only available via the chat API. However, our API expects a CompletionResponse.
            response = self._get_client().chat.completions.create(**params)
            choice = response.choices[0]
            message = choice.message
            # If message.content is returned as a list of tokens, join them into a string.
            content = message.content if not isinstance(
                message.content, list) else "".join(message.content)
            return CompletionResponse(
                content=content,
                stop_reason=
                "end_of_message",  # ***** HACK: Hard-coded stop_reason because the chat API doesn't return one.
                logprobs=None,
            )
        else:
            # ***** HACK: For non-structured outputs, ensure we use the completions endpoint.
            # _get_params may include a "messages" key due to our unified parameter builder.
            # We remove "messages" and instead set a "prompt" since the completions endpoint expects it.
            prompt_str = await completion_request_to_prompt(request)
            if "messages" in params:
                del params["messages"]
            params["prompt"] = prompt_str
            response = self._get_client().completions.create(**params)
            result = process_completion_response(response)
            # Join tokenized content if needed.
            if isinstance(result.content, list):
                result.content = "".join(result.content)
            return result

    async def _stream_completion(self,
                                 request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _to_async_generator():
            # ***** HACK: For streaming structured outputs, use the chat completions endpoint.
            # Otherwise, use the regular completions endpoint.
            if request.response_format is not None:
                stream = self._get_client().chat.completions.create(**params)
            else:
                stream = self._get_client().completions.create(**params)
            for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        if request.response_format is not None:
            async for chunk in process_chat_completion_stream_response(
                    stream, request):
                yield chunk
        else:
            async for chunk in process_completion_stream_response(stream):
                yield chunk

    #
    # CHAT COMPLETION
    #

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
        """
        For "chat completion" style requests.
        """
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
            self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        # Use the chat completions endpoint if "messages" key is present.
        if "messages" in params:
            response = self._get_client().chat.completions.create(**params)
        else:
            response = self._get_client().completions.create(**params)
        result = process_chat_completion_response(response, request)
        # ***** HACK: Sometimes the returned content is tokenized as a list.
        # We join the tokens into a single string to produce a unified output.
        if request.response_format is not None:
            if isinstance(result.completion_message, dict):
                content = result.completion_message.get("content")
                if isinstance(content, list):
                    result.completion_message["content"] = "".join(content)
            else:
                if isinstance(result.completion_message.content, list):
                    updated_msg = result.completion_message.copy(update={
                        "content":
                        "".join(result.completion_message.content)
                    })
                    result = result.copy(
                        update={"completion_message": updated_msg})
        return result

    async def _stream_chat_completion(
            self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _to_async_generator():
            # ***** HACK: Use the chat completions endpoint if "messages" key is present.
            if "messages" in params:
                stream = self._get_client().chat.completions.create(**params)
            else:
                stream = self._get_client().completions.create(**params)
            for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(
                stream, request):
            yield chunk

    #
    # HELPER METHODS
    #

    async def _get_params(
            self, request: Union[ChatCompletionRequest,
                                 CompletionRequest]) -> dict:
        """
        Build a unified set of parameters for both chat and non-chat requests.
        When a structured output is specified (response_format is not None), we force
        the use of a "messages" array even for CompletionRequests.
        """
        input_dict = {}
        media_present = request_has_media(request)
        llama_model = self.get_llama_model(request.model)
        if request.response_format is not None:
            if isinstance(request, ChatCompletionRequest):
                input_dict["messages"] = [
                    await convert_message_to_openai_dict(m)
                    for m in request.messages
                ]
            else:
                # ***** HACK: For CompletionRequests with structured output,
                # we simulate a chat conversation by wrapping the prompt as a single user message.
                prompt_str = await completion_request_to_prompt(request)
                input_dict["messages"] = [{
                    "role": "user",
                    "content": prompt_str
                }]
        else:
            if isinstance(request, ChatCompletionRequest):
                if media_present or not llama_model:
                    input_dict["messages"] = [
                        await convert_message_to_openai_dict(m)
                        for m in request.messages
                    ]
                else:
                    input_dict[
                        "prompt"] = await chat_completion_request_to_prompt(
                            request, llama_model)
            else:
                input_dict["prompt"] = await completion_request_to_prompt(
                    request)
        params = {
            "model":
            request.model,
            **input_dict,
            "stream":
            request.stream,
            **self._build_options(request.sampling_params, request.logprobs, request.response_format),
        }
        return params

    def _build_options(
        self,
        sampling_params: Optional[SamplingParams],
        logprobs: Optional[LogProbConfig],
        fmt: Optional[ResponseFormat],
    ) -> dict:
        """
        Build additional options such as sampling parameters and logprobs.
        Also translates our response_format into the format expected by CentML's API.
        """
        options = get_sampling_options(sampling_params)
        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "schema",
                        "schema": fmt.json_schema
                    },
                }
            elif fmt.type == ResponseFormatType.grammar.value:
                raise NotImplementedError(
                    "Grammar response format not supported yet")
            else:
                raise ValueError(f"Unknown response format {fmt.type}")
        if logprobs and logprobs.top_k:
            options["logprobs"] = logprobs.top_k
        return options

    #
    # EMBEDDINGS
    #

    async def embeddings(
        self,
        task_type: str,
        model_id: str,
        text_truncation: Optional[str],
        output_dimension: Optional[int],
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        # this will come in future updates
        model = await self.model_store.get_model(model_id)
        assert all(not content_has_media(c) for c in contents), (
            "CentML does not support media for embeddings")
        resp = self._get_client().embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(c) for c in contents],
        )
        embeddings = [item.embedding for item in resp.data]
        return EmbeddingsResponse(embeddings=embeddings)
