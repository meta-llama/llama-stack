# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import List, Optional, Protocol, runtime_checkable

from llama_stack.apis.inference import (
    BatchChatCompletionResponse,
    BatchCompletionResponse,
    InterleavedContent,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    ToolConfig,
    ToolDefinition,
)
from llama_stack.schema_utils import webmethod


@runtime_checkable
class BatchInference(Protocol):
    @webmethod(route="/batch-inference/completion-inline", method="POST")
    async def batch_completion_inline(
        self,
        model: str,
        content_batch: List[InterleavedContent],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ) -> BatchCompletionResponse: ...

    @webmethod(route="/batch-inference/chat-completion-inline", method="POST")
    async def batch_chat_completion_inline(
        self,
        model: str,
        messages_batch: List[List[Message]],
        sampling_params: Optional[SamplingParams] = None,
        tools: Optional[List[ToolDefinition]] = list,
        tool_config: Optional[ToolConfig] = None,
        response_format: Optional[ResponseFormat] = None,
        logprobs: Optional[LogProbConfig] = None,
    ) -> BatchChatCompletionResponse: ...
