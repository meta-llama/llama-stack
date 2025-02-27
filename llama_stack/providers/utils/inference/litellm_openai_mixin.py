# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, AsyncIterator, List, Optional, Union

import litellm

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
    JsonSchemaResponseFormat,
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
from llama_stack.apis.models.models import Model
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict_new,
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
    convert_tooldef_to_openai_tool,
    get_sampling_options,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    interleaved_content_as_str,
)


class LiteLLMOpenAIMixin(
    ModelRegistryHelper,
    Inference,
):
    def __init__(self, model_entries) -> None:
        self.model_entries = model_entries
        ModelRegistryHelper.__init__(self, model_entries)

    async def register_model(self, model: Model) -> Model:
        model_id = self.get_provider_model_id(model.provider_resource_id)
        if model_id is None:
            raise ValueError(f"Unsupported model: {model.provider_resource_id}")
        return model

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError("LiteLLM does not support completion requests")

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
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
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

        params = await self._get_params(request)
        # unfortunately, we need to use synchronous litellm.completion here because litellm
        # caches various httpx.client objects in a non-eventloop aware manner

        from rich.pretty import pprint

        pprint(params)
        response = litellm.completion(**params)
        if stream:
            return self._stream_chat_completion(response)
        else:
            return convert_openai_chat_completion_choice(response.choices[0])

    async def _stream_chat_completion(
        self, response: litellm.ModelResponse
    ) -> AsyncIterator[ChatCompletionResponseStreamChunk]:
        async def _stream_generator():
            for chunk in response:
                yield chunk

        async for chunk in convert_openai_chat_completion_stream(
            _stream_generator(), enable_incremental_tool_calls=True
        ):
            yield chunk

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        input_dict = {}

        input_dict["messages"] = [await convert_message_to_openai_dict_new(m) for m in request.messages]
        if fmt := request.response_format:
            if not isinstance(fmt, JsonSchemaResponseFormat):
                raise ValueError(
                    f"Unsupported response format: {type(fmt)}. Only JsonSchemaResponseFormat is supported."
                )

            fmt = fmt.json_schema
            name = fmt["title"]
            del fmt["title"]
            fmt["additionalProperties"] = False
            input_dict["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": fmt,
                    "strict": True,
                },
            }
        if request.tools:
            input_dict["tools"] = [convert_tooldef_to_openai_tool(tool) for tool in request.tools]
            if request.tool_config.tool_choice:
                input_dict["tool_choice"] = request.tool_config.tool_choice.value

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **get_sampling_options(request.sampling_params),
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

        response = litellm.embedding(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
        )

        embeddings = [data["embedding"] for data in response["data"]]
        return EmbeddingsResponse(embeddings=embeddings)
