# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel, Field

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403


@json_schema_type
class BatchCompletionRequest(BaseModel):
    model: str
    content_batch: List[InterleavedTextMedia]
    sampling_params: Optional[SamplingParams] = SamplingParams()
    logprobs: Optional[LogProbConfig] = None


@json_schema_type
class BatchCompletionResponse(BaseModel):
    completion_message_batch: List[CompletionMessage]


@json_schema_type
class BatchChatCompletionRequest(BaseModel):
    model: str
    messages_batch: List[List[Message]]
    sampling_params: Optional[SamplingParams] = SamplingParams()

    # zero-shot tool definitions as input to the model
    tools: Optional[List[ToolDefinition]] = Field(default_factory=list)
    tool_choice: Optional[ToolChoice] = Field(default=ToolChoice.auto)
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )
    logprobs: Optional[LogProbConfig] = None


@json_schema_type
class BatchChatCompletionResponse(BaseModel):
    completion_message_batch: List[CompletionMessage]


@runtime_checkable
class BatchInference(Protocol):
    @webmethod(route="/batch_inference/completion")
    async def batch_completion(
        self,
        model: str,
        content_batch: List[InterleavedTextMedia],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        logprobs: Optional[LogProbConfig] = None,
    ) -> BatchCompletionResponse: ...

    @webmethod(route="/batch_inference/chat_completion")
    async def batch_chat_completion(
        self,
        model: str,
        messages_batch: List[List[Message]],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        # zero-shot tool definitions as input to the model
        tools: Optional[List[ToolDefinition]] = list,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        logprobs: Optional[LogProbConfig] = None,
    ) -> BatchChatCompletionResponse: ...
