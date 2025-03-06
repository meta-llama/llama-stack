# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel

from llama_stack.apis.inference import (
    ChatCompletionResponse,
    CompletionResponse,
    InterleavedContent,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.schema_utils import json_schema_type, webmethod


@json_schema_type
class BatchCompletionResponse(BaseModel):
    batch: List[CompletionResponse]


@json_schema_type
class BatchChatCompletionResponse(BaseModel):
    batch: List[ChatCompletionResponse]


@runtime_checkable
class BatchInference(Protocol):
    @webmethod(route="/batch-inference/completion", method="POST")
    async def batch_completion(
        self,
        model: str,
        content_batch: List[InterleavedContent],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ) -> BatchCompletionResponse: ...

    @webmethod(route="/batch-inference/chat-completion", method="POST")
    async def batch_chat_completion(
        self,
        model: str,
        messages_batch: List[List[Message]],
        sampling_params: Optional[SamplingParams] = None,
        # zero-shot tool definitions as input to the model
        tools: Optional[List[ToolDefinition]] = list,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ) -> BatchChatCompletionResponse: ...
