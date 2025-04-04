# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import AsyncGenerator, List, Optional

from openai import AsyncOpenAI, BadRequestError

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionRequest,
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
from llama_stack.apis.models import Model
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
)

from .models import model_entries
from .openai_utils import (
    convert_chat_completion_request,
    convert_completion_request,
    convert_openai_completion_choice,
    convert_openai_completion_stream,
)

logger = get_logger(name=__name__, category="inference")


class RamalamaInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, url: str) -> None:
        self.register_helper = ModelRegistryHelper(model_entries)
        self.url = url

    async def initialize(self) -> None:
        logger.info(f"checking connectivity to Ramalama at `{self.url}`...")
        self.client = AsyncOpenAI(base_url=self.url, api_key="NO KEY")

    async def shutdown(self) -> None:
        pass

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = convert_completion_request(
            request=CompletionRequest(
                model=model.provider_resource_id,
                content=content,
                sampling_params=sampling_params,
                response_format=response_format,
                stream=stream,
                logprobs=logprobs,
            )
        )

        response = await self.client.completions.create(**request)
        if stream:
            return convert_openai_completion_stream(response)
        else:
            # we pass n=1 to get only one completion
            return convert_openai_completion_choice(response.choices[0])

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
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        request = await convert_chat_completion_request(
            request=ChatCompletionRequest(
                model=model.provider_resource_id,
                messages=messages,
                sampling_params=sampling_params,
                tools=tools or [],
                stream=stream,
                logprobs=logprobs,
                response_format=response_format,
                tool_config=tool_config,
            ),
            n=1,
        )
        s = await self.client.chat.completions.create(**request)
        if stream:
            return convert_openai_chat_completion_stream(s, enable_incremental_tool_calls=False)
        else:
            # we pass n=1 to get only one completion
            return convert_openai_chat_completion_choice(s.choices[0])

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        flat_contents = [content.text if isinstance(content, TextContentItem) else content for content in contents]
        input = [content.text if isinstance(content, TextContentItem) else content for content in flat_contents]
        model = self.get_provider_model_id(model_id)

        extra_body = {}

        if text_truncation is not None:
            text_truncation_options = {
                TextTruncation.none: "NONE",
                TextTruncation.end: "END",
                TextTruncation.start: "START",
            }
            extra_body["truncate"] = text_truncation_options[text_truncation]

        if output_dimension is not None:
            extra_body["dimensions"] = output_dimension

        if task_type is not None:
            task_type_options = {
                EmbeddingTaskType.document: "passage",
                EmbeddingTaskType.query: "query",
            }
            extra_body["input_type"] = task_type_options[task_type]

        try:
            response = await self._client.embeddings.create(
                model=model,
                input=input,
                extra_body=extra_body,
            )
        except BadRequestError as e:
            raise ValueError(f"Failed to get embeddings: {e}") from e

        return EmbeddingsResponse(embeddings=[embedding.embedding for embedding in response.data])

    async def register_model(self, model: Model) -> Model:
        model = await self.register_helper.register_model(model)
        res = await self.client.models.list()
        available_models = [m.id async for m in res]
        if model.provider_resource_id not in available_models:
            raise ValueError(
                f"Model {model.provider_resource_id} is not being served by vLLM. "
                f"Available models: {', '.join(available_models)}"
            )
        return model
