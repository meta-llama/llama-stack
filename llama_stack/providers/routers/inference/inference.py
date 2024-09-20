# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, AsyncGenerator, Dict, List, Tuple

from llama_stack.distribution.datatypes import Api
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.registry.inference import available_providers


class InferenceRouterImpl(Inference):
    """Routes to an provider based on the memory bank type"""

    def __init__(
        self,
        inner_impls: List[Tuple[str, Any]],
        deps: List[Api],
    ) -> None:
        self.inner_impls = inner_impls
        self.deps = deps
        print("INIT INFERENCE ROUTER!")

        # self.providers = {}
        # for routing_key, provider_impl in inner_impls:
        #     self.providers[routing_key] = provider_impl

    async def initialize(self) -> None:
        pass

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
