# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional, Union

from openai import AsyncOpenAI, BadRequestError
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)

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
from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAICompletion,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.apis.models import Model
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_chat_completion_request,
    convert_completion_request,
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
    convert_openai_completion_choice,
    convert_openai_completion_stream,
    prepare_openai_completion_params,
)

from .models import model_entries

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
        # Ramalama handles paths on MacOS and Linux differently
        if (model.provider_resource_id.split("/")[-1] not in available_models) and (
            model.provider_resource_id not in available_models
        ):
            raise ValueError(
                f"Model {model.provider_resource_id} is not being served by Ramalama. "
                f"Available models: {', '.join(available_models)}"
            )
        return model

    async def openai_completion(
        self,
        model: str,
        prompt: Union[str, List[str], List[int], List[List[int]]],
        best_of: Optional[int] = None,
        echo: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        guided_choice: Optional[List[str]] = None,
        prompt_logprobs: Optional[int] = None,
    ) -> OpenAICompletion:
        model_obj = await self.model_store.get_model(model)
        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            user=user,
        )
        return await self.client.completions.create(**params)  # type: ignore

    async def openai_chat_completion(
        self,
        model: str,
        messages: List[OpenAIMessageParam],
        frequency_penalty: Optional[float] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[OpenAIResponseFormatParam] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_logprobs: Optional[int] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
    ) -> Union[OpenAIChatCompletion, AsyncIterator[OpenAIChatCompletionChunk]]:
        model_obj = await self.model_store.get_model(model)
        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
        )
        return await self.client.chat.completions.create(**params)  # type: ignore

    async def batch_completion(
        self,
        model_id: str,
        content_batch: List[InterleavedContent],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ):
        raise NotImplementedError("Batch completion is not supported for Ramalama")

    async def batch_chat_completion(
        self,
        model_id: str,
        messages_batch: List[List[Message]],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_config: Optional[ToolConfig] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ):
        raise NotImplementedError("Batch chat completion is not supported for Ramalama")
