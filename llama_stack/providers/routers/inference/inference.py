# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, AsyncGenerator, Dict, List, Tuple

from llama_stack.distribution.datatypes import Api
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.models import Models

from llama_stack.distribution.datatypes import GenericProviderConfig
from llama_stack.distribution.distribution import api_providers
from llama_stack.distribution.utils.dynamic import instantiate_provider
from llama_stack.providers.impls.builtin.models.models import BuiltinModelsImpl
from llama_stack.providers.registry.inference import available_providers
from termcolor import cprint


class InferenceRouterImpl(Inference):
    """Routes to an provider based on the memory bank type"""

    def __init__(
        self,
        models_api: Models,
    ) -> None:
        # map of model_id to provider impl
        self.providers = {}
        self.models_api = models_api

    async def initialize(self) -> None:
        inference_providers = api_providers()[Api.inference]

        models_list_response = await self.models_api.list_models()
        for model_spec in models_list_response.models_list:

            if model_spec.api != Api.inference.value:
                continue

            if model_spec.provider_id not in inference_providers:
                raise ValueError(
                    f"provider_id {model_spec.provider_id} is not available for inference. Please check run.yaml config spec to define a valid provider"
                )
            impl = await instantiate_provider(
                inference_providers[model_spec.provider_id],
                deps=[],
                provider_config=GenericProviderConfig(
                    provider_id=model_spec.provider_id,
                    config=model_spec.provider_config,
                ),
            )
            cprint(f"impl={impl}", "blue")
            # look up and initialize provider implementations for each model
            core_model_id = model_spec.llama_model_metadata.core_model_id

    async def shutdown(self) -> None:
        pass
        # for p in self.providers.values():
        #     await p.shutdown()

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        # zero-shot tool definitions as input to the model
        tools: Optional[List[ToolDefinition]] = list,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        print("router chat_completion")
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.progress,
                delta="router chat completion",
            )
        )
        # async for chunk in self.providers[model].chat_completion(
        #     model=model,
        #     messages=messages,
        #     sampling_params=sampling_params,
        #     tools=tools,
        #     tool_choice=tool_choice,
        #     tool_prompt_format=tool_prompt_format,
        #     stream=stream,
        #     logprobs=logprobs,
        # ):
        #     yield chunk
