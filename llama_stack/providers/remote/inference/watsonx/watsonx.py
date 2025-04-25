# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator, List, Optional, Union

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams

from llama_stack.apis.common.content_types import InterleavedContent, InterleavedContentItem
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
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
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
    request_has_media,
)

from . import WatsonXConfig
from .models import MODEL_ENTRIES


class WatsonXInferenceAdapter(Inference, ModelRegistryHelper):
    def __init__(self, config: WatsonXConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ENTRIES)

        print(f"Initializing watsonx InferenceAdapter({config.url})...")

        self._config = config

        self._project_id = self._config.project_id

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
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

    def _get_client(self, model_id) -> Model:
        config_api_key = self._config.api_key.get_secret_value() if self._config.api_key else None
        config_url = self._config.url
        project_id = self._config.project_id
        credentials = {"url": config_url, "apikey": config_api_key}

        return Model(model_id=model_id, credentials=credentials, project_id=project_id)

    async def _nonstream_completion(self, request: CompletionRequest) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = self._get_client(request.model).generate(**params)
        choices = []
        if "results" in r:
            for result in r["results"]:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=result["stop_reason"] if result["stop_reason"] else None,
                    text=result["generated_text"],
                )
                choices.append(choice)
        response = OpenAICompatCompletionResponse(
            choices=choices,
        )
        return process_completion_response(response)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _generate_and_convert_to_openai_compat():
            s = self._get_client(request.model).generate_text_stream(**params)
            for chunk in s:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=None,
                    text=chunk,
                )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
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
        params = await self._get_params(request)
        r = self._get_client(request.model).generate(**params)
        choices = []
        if "results" in r:
            for result in r["results"]:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=result["stop_reason"] if result["stop_reason"] else None,
                    text=result["generated_text"],
                )
                choices.append(choice)
        response = OpenAICompatCompletionResponse(
            choices=choices,
        )
        return process_chat_completion_response(response, request)

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)
        model_id = request.model

        # if we shift to TogetherAsyncClient, we won't need this wrapper
        async def _to_async_generator():
            s = self._get_client(model_id).generate_text_stream(**params)
            for chunk in s:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=None,
                    text=chunk,
                )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def _get_params(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> dict:
        input_dict = {"params": {}}
        media_present = request_has_media(request)
        llama_model = self.get_llama_model(request.model)
        if isinstance(request, ChatCompletionRequest):
            input_dict["prompt"] = await chat_completion_request_to_prompt(request, llama_model)
        else:
            assert not media_present, "Together does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)
        if request.sampling_params:
            if request.sampling_params.strategy:
                input_dict["params"][GenParams.DECODING_METHOD] = request.sampling_params.strategy.type
            if request.sampling_params.max_tokens:
                input_dict["params"][GenParams.MAX_NEW_TOKENS] = request.sampling_params.max_tokens
            if request.sampling_params.repetition_penalty:
                input_dict["params"][GenParams.REPETITION_PENALTY] = request.sampling_params.repetition_penalty
            if request.sampling_params.additional_params.get("top_p"):
                input_dict["params"][GenParams.TOP_P] = request.sampling_params.additional_params["top_p"]
            if request.sampling_params.additional_params.get("top_k"):
                input_dict["params"][GenParams.TOP_K] = request.sampling_params.additional_params["top_k"]
            if request.sampling_params.additional_params.get("temperature"):
                input_dict["params"][GenParams.TEMPERATURE] = request.sampling_params.additional_params["temperature"]
            if request.sampling_params.additional_params.get("length_penalty"):
                input_dict["params"][GenParams.LENGTH_PENALTY] = request.sampling_params.additional_params[
                    "length_penalty"
                ]
            if request.sampling_params.additional_params.get("random_seed"):
                input_dict["params"][GenParams.RANDOM_SEED] = request.sampling_params.additional_params["random_seed"]
            if request.sampling_params.additional_params.get("min_new_tokens"):
                input_dict["params"][GenParams.MIN_NEW_TOKENS] = request.sampling_params.additional_params[
                    "min_new_tokens"
                ]
            if request.sampling_params.additional_params.get("stop_sequences"):
                input_dict["params"][GenParams.STOP_SEQUENCES] = request.sampling_params.additional_params[
                    "stop_sequences"
                ]
            if request.sampling_params.additional_params.get("time_limit"):
                input_dict["params"][GenParams.TIME_LIMIT] = request.sampling_params.additional_params["time_limit"]
            if request.sampling_params.additional_params.get("truncate_input_tokens"):
                input_dict["params"][GenParams.TRUNCATE_INPUT_TOKENS] = request.sampling_params.additional_params[
                    "truncate_input_tokens"
                ]
            if request.sampling_params.additional_params.get("return_options"):
                input_dict["params"][GenParams.RETURN_OPTIONS] = request.sampling_params.additional_params[
                    "return_options"
                ]

        params = {
            **input_dict,
        }
        return params

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        pass
