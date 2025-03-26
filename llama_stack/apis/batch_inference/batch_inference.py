# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Protocol, runtime_checkable

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
    batch: list[CompletionResponse]


@json_schema_type
class BatchChatCompletionResponse(BaseModel):
    batch: list[ChatCompletionResponse]


@runtime_checkable
class BatchInference(Protocol):
    @webmethod(route="/batch-inference/completion", method="POST")
    async def batch_completion(
        self,
        model: str,
        content_batch: list[InterleavedContent],
        sampling_params: SamplingParams | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> BatchCompletionResponse: ...

    @webmethod(route="/batch-inference/chat-completion", method="POST")
    async def batch_chat_completion(
        self,
        model: str,
        messages_batch: list[list[Message]],
        sampling_params: SamplingParams | None = None,
        # zero-shot tool definitions as input to the model
        tools: list[ToolDefinition] | None = list,
        tool_choice: ToolChoice | None = ToolChoice.auto,
        tool_prompt_format: ToolPromptFormat | None = None,
        response_format: ResponseFormat | None = None,
        logprobs: LogProbConfig | None = None,
    ) -> BatchChatCompletionResponse: ...
