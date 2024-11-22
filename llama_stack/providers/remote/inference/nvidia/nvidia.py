# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from typing import AsyncIterator, List, Optional, Union

from llama_models.datatypes import SamplingParams
from llama_models.llama3.api.datatypes import (
    InterleavedTextMedia,
    Message,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_models.sku_list import CoreModelId
from openai import APIConnectionError, AsyncOpenAI

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    Inference,
    LogProbConfig,
    ResponseFormat,
)
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)

from . import NVIDIAConfig
from .openai_utils import (
    convert_chat_completion_request,
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
)
from .utils import _is_nvidia_hosted, check_health

_MODEL_ALIASES = [
    build_model_alias(
        "meta/llama3-8b-instruct",
        CoreModelId.llama3_8b_instruct.value,
    ),
    build_model_alias(
        "meta/llama3-70b-instruct",
        CoreModelId.llama3_70b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.1-8b-instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.1-70b-instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.1-405b-instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.2-1b-instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.2-3b-instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.2-11b-vision-instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "meta/llama-3.2-90b-vision-instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    # TODO(mf): how do we handle Nemotron models?
    # "Llama3.1-Nemotron-51B-Instruct" -> "meta/llama-3.1-nemotron-51b-instruct",
]


class NVIDIAInferenceAdapter(Inference, ModelRegistryHelper):
    def __init__(self, config: NVIDIAConfig) -> None:
        # TODO(mf): filter by available models
        ModelRegistryHelper.__init__(self, model_aliases=_MODEL_ALIASES)

        print(f"Initializing NVIDIAInferenceAdapter({config.url})...")

        if _is_nvidia_hosted(config):
            if not config.api_key:
                raise RuntimeError(
                    "API key is required for hosted NVIDIA NIM. "
                    "Either provide an API key or use a self-hosted NIM."
                )
        # elif self._config.api_key:
        #
        # we don't raise this warning because a user may have deployed their
        # self-hosted NIM with an API key requirement.
        #
        #     warnings.warn(
        #         "API key is not required for self-hosted NVIDIA NIM. "
        #         "Consider removing the api_key from the configuration."
        #     )

        self._config = config
        # make sure the client lives longer than any async calls
        self._client = AsyncOpenAI(
            base_url=f"{self._config.url}/v1",
            api_key=self._config.api_key or "NO KEY",
            timeout=self._config.timeout,
        )

    def completion(
        self,
        model_id: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        raise NotImplementedError()

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[
            ToolPromptFormat
        ] = None,  # API default is ToolPromptFormat.json, we default to None to detect user input
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:
        if tool_prompt_format:
            warnings.warn("tool_prompt_format is not supported by NVIDIA NIM, ignoring")

        await check_health(self._config)  # this raises errors

        request = convert_chat_completion_request(
            request=ChatCompletionRequest(
                model=self.get_provider_model_id(model_id),
                messages=messages,
                sampling_params=sampling_params,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                tool_prompt_format=tool_prompt_format,
                stream=stream,
                logprobs=logprobs,
            ),
            n=1,
        )

        try:
            response = await self._client.chat.completions.create(**request)
        except APIConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to NVIDIA NIM at {self._config.url}: {e}"
            ) from e

        if stream:
            return convert_openai_chat_completion_stream(response)
        else:
            # we pass n=1 to get only one completion
            return convert_openai_chat_completion_choice(response.choices[0])
