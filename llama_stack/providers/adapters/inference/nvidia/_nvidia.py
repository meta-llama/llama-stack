# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import warnings
from typing import Dict, List, Optional, Union

import httpx
from llama_models.datatypes import SamplingParams
from llama_models.llama3.api.datatypes import (
    InterleavedTextMedia,
    Message,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_models.sku_list import CoreModelId

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseStreamChunk,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    Inference,
    LogProbConfig,
    ModelDef,
    ResponseFormat,
)

from ._config import NVIDIAConfig
from ._utils import check_health, convert_chat_completion_request, parse_completion

SUPPORTED_MODELS: Dict[CoreModelId, str] = {
    CoreModelId.llama3_8b_instruct: "meta/llama3-8b-instruct",
    CoreModelId.llama3_70b_instruct: "meta/llama3-70b-instruct",
    CoreModelId.llama3_1_8b_instruct: "meta/llama-3.1-8b-instruct",
    CoreModelId.llama3_1_70b_instruct: "meta/llama-3.1-70b-instruct",
    CoreModelId.llama3_1_405b_instruct: "meta/llama-3.1-405b-instruct",
    # TODO(mf): how do we handle Nemotron models?
    # "Llama3.1-Nemotron-51B-Instruct": "meta/llama-3.1-nemotron-51b-instruct",
    CoreModelId.llama3_2_1b_instruct: "meta/llama-3.2-1b-instruct",
    CoreModelId.llama3_2_3b_instruct: "meta/llama-3.2-3b-instruct",
    CoreModelId.llama3_2_11b_vision_instruct: "meta/llama-3.2-11b-vision-instruct",
    CoreModelId.llama3_2_90b_vision_instruct: "meta/llama-3.2-90b-vision-instruct",
}


class NVIDIAInferenceAdapter(Inference):
    def __init__(self, config: NVIDIAConfig) -> None:

        print(f"Initializing NVIDIAInferenceAdapter({config.base_url})...")

        if config.is_hosted:
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

    @property
    def _headers(self) -> dict:
        return {
            b"User-Agent": b"llama-stack: nvidia-inference-adapter",
            **(
                {b"Authorization": f"Bearer {self._config.api_key}"}
                if self._config.api_key
                else {}
            ),
        }

    async def list_models(self) -> List[ModelDef]:
        # TODO(mf): filter by available models
        return [
            ModelDef(identifier=model, llama_model=id_)
            for model, id_ in SUPPORTED_MODELS.items()
        ]

    def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
        raise NotImplementedError()

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()

    async def chat_completion(
        self,
        model: str,
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
    ) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]:
        if tool_prompt_format:
            warnings.warn("tool_prompt_format is not supported by NVIDIA NIM, ignoring")

        if stream:
            raise ValueError("Streamed completions are not supported")

        await check_health(self._config)  # this raises errors

        request = ChatCompletionRequest(
            model=SUPPORTED_MODELS[CoreModelId(model)],
            messages=messages,
            sampling_params=sampling_params,
            tools=tools,
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            try:
                response = await client.post(
                    f"{self._config.base_url}/v1/chat/completions",
                    headers=self._headers,
                    json=convert_chat_completion_request(request, n=1),
                )
            except httpx.ReadTimeout as e:
                raise TimeoutError(
                    f"Request timed out. timeout set to {self._config.timeout}. Use `llama stack configure ...` to adjust it."
                ) from e

            if response.status_code == 401:
                raise PermissionError(
                    "Unauthorized. Please check your API key, reconfigure, and try again."
                )

            if response.status_code == 400:
                raise ValueError(
                    f"Bad request. Please check the request and try again. Detail: {response.text}"
                )

            if response.status_code == 404:
                raise ValueError(
                    "Model not found. Please check the model name and try again."
                )

            assert (
                response.status_code == 200
            ), f"Failed to get completion: {response.text}"

            # we pass n=1 to get only one completion
            return parse_completion(response.json()["choices"][0])
