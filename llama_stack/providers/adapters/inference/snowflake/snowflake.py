# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import AsyncGenerator

import httpx

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
)

from .config import SnowflakeImplConfig


SNOWFLAKE_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "llama3.1-8b",
    "Llama3.1-70B-Instruct": "llama3.1-70b",
    "Llama3.1-405B-Instruct": "llama3.1-405b",
}


class SnowflakeInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):

    def __init__(self, config: SnowflakeImplConfig) -> None:
        ModelRegistryHelper.__init__(
            self, stack_to_provider_models_map=SNOWFLAKE_SUPPORTED_MODELS
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

    def _get_cortex_headers(
        self,
    ):
        snowflake_api_key = None
        if self.config.api_key is not None:
            snowflake_api_key = self.config.api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.snowflake_api_key:
                raise ValueError(
                    'Pass Snowflake API Key in the header X-LlamaStack-ProviderData as { "snowflake_api_key": <your api key>}'
                )
            snowflake_api_key = provider_data.snowflake_api_key

        headers = {
            "Accept": "text/stream",
            "Content-Type": "application/json",
            "Authorization": f'Snowflake Token="{snowflake_api_key}"',
        }

        return headers

    def _get_cortex_client(self, timeout=30, concurrent_limit=1000):

        client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=concurrent_limit,
                max_keepalive_connections=concurrent_limit,
            ),
        )

        return client

    def _get_cortex_async_client(self, timeout=30, concurrent_limit=1000):

        client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=concurrent_limit,
                max_keepalive_connections=concurrent_limit,
            ),
        )

        return client

    async def _nonstream_completion(
        self, request: CompletionRequest
    ) -> ChatCompletionResponse:
        params = self._get_params_for_completion(request)
        r = self._get_cortex_client().post(**params)
        return process_completion_response(
            r, self.formatter
        )  # TODO VALIDATE COMPLETION PROCESSOR

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = self._get_params_for_completion(request)

        async def _to_async_generator():
            s = self._get_cortex_client().post(**params)
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
        r = self._get_cortex_client().post(**params)
        return self._process_nonstream_snowflake_response(r.text)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        params = self._get_params(request)

        async def _to_async_generator():
            async with self._get_cortex_async_client() as client:
                async with client.stream("POST", **params) as response:
                    async for line in response.aiter_lines():
                        if line.strip():  # Check if line is not empty
                            yield line

        stream = _to_async_generator()

        async for chunk in stream:
            clean_chunk = self._process_snowflake_stream_response(chunk)
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.progress,
                    delta=clean_chunk,
                    stop_reason=None,
                )
            )

    def _get_params(self, request: ChatCompletionRequest) -> dict:
        return {
            "url": self._get_cortex_url(),
            "headers": self._get_cortex_headers(),
            "json": {
                "model": self.map_to_provider_model(request.model),
                "messages": [
                    {
                        "content": chat_completion_request_to_prompt(
                            request, self.formatter
                        )
                    }
                ],
            },
        }

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()

    def _process_nonstream_snowflake_response(self, response_str):

        json_objects = response_str.split("\ndata: ")
        json_list = []

        # Iterate over each JSON object
        for obj in json_objects:
            obj = obj.strip()
            if obj:
                # Remove the 'data: ' prefix if it exists
                if obj.startswith("data: "):
                    obj = obj[6:]
                # Load the JSON object into a Python dictionary
                json_dict = json.loads(obj, strict=False)
                # Append the JSON dictionary to the list
                json_list.append(json_dict)

        completion = ""
        choices = {}
        for chunk in json_list:
            choices = chunk["choices"][0]

            if "content" in choices["delta"].keys():
                completion += choices["delta"]["content"]

        return completion

    def _process_snowflake_stream_response(self, response_str):
        if not response_str.startswith("data: "):
            return ""

        try:
            json_dict = json.loads(response_str[6:])
            return json_dict["choices"][0]["delta"].get("content", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            return ""

    def _get_cortex_url(self):
        account_id = self.config.account
        cortex_endpoint = f"https://{account_id}.snowflakecomputing.com/api/v2/cortex/inference:complete"
        return cortex_endpoint
