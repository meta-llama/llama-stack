# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

from botocore.client import BaseClient

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.providers.remote.inference.bedrock.config import BedrockConfig
from llama_stack.providers.utils.bedrock.client import create_bedrock_client
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    get_sampling_strategy_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
)

from .models import MODEL_ENTRIES


class BedrockInferenceAdapter(ModelRegistryHelper, Inference):
    def __init__(self, config: BedrockConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES)
        self._config = config

        self._client = create_bedrock_client(config)

    @property
    def client(self) -> BaseClient:
        return self._client

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        self.client.close()

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        if sampling_params is None:
            sampling_params = SamplingParams()
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
        params = await self._get_params_for_chat_completion(request)
        res = self.client.invoke_model(**params)
        chunk = next(res["body"])
        result = json.loads(chunk.decode("utf-8"))

        choice = OpenAICompatCompletionChoice(
            finish_reason=result["stop_reason"],
            text=result["generation"],
        )

        response = OpenAICompatCompletionResponse(choices=[choice])
        return process_chat_completion_response(response, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params_for_chat_completion(request)
        res = self.client.invoke_model_with_response_stream(**params)
        event_stream = res["body"]

        async def _generate_and_convert_to_openai_compat():
            for chunk in event_stream:
                chunk = chunk["chunk"]["bytes"]
                result = json.loads(chunk.decode("utf-8"))
                choice = OpenAICompatCompletionChoice(
                    finish_reason=result["stop_reason"],
                    text=result["generation"],
                )
                yield OpenAICompatCompletionResponse(choices=[choice])

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params_for_chat_completion(self, request: ChatCompletionRequest) -> Dict:
        bedrock_model = request.model

        sampling_params = request.sampling_params
        options = get_sampling_strategy_options(sampling_params)

        if sampling_params.max_tokens:
            options["max_gen_len"] = sampling_params.max_tokens
        if sampling_params.repetition_penalty > 0:
            options["repetition_penalty"] = sampling_params.repetition_penalty

        prompt = await chat_completion_request_to_prompt(request, self.get_llama_model(request.model))
        return {
            "modelId": bedrock_model,
            "body": json.dumps(
                {
                    "prompt": prompt,
                    **options,
                }
            ),
        }

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)
        embeddings = []
        for content in contents:
            assert not content_has_media(content), "Bedrock does not support media for embeddings"
            input_text = interleaved_content_as_str(content)
            input_body = {"inputText": input_text}
            body = json.dumps(input_body)
            response = self.client.invoke_model(
                body=body,
                modelId=model.provider_resource_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            embeddings.append(response_body.get("embedding"))
        return EmbeddingsResponse(embeddings=embeddings)
