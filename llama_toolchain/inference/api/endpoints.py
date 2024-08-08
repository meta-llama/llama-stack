# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .datatypes import *  # noqa: F403
from typing import Optional, Protocol

# this dependency is annoying and we need a forked up version anyway
from llama_models.schema_utils import webmethod


@json_schema_type
class CompletionRequest(BaseModel):
    model: str
    content: InterleavedTextAttachment
    sampling_params: Optional[SamplingParams] = SamplingParams()

    stream: Optional[bool] = False
    logprobs: Optional[LogProbConfig] = None
    quantization_config: Optional[QuantizationConfig] = None


@json_schema_type
class CompletionResponse(BaseModel):
    completion_message: CompletionMessage
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class CompletionResponseStreamChunk(BaseModel):
    """streamed completion response."""

    delta: str
    stop_reason: Optional[StopReason] = None
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class BatchCompletionRequest(BaseModel):
    model: str
    content_batch: List[InterleavedTextAttachment]
    sampling_params: Optional[SamplingParams] = SamplingParams()
    logprobs: Optional[LogProbConfig] = None
    quantization_config: Optional[QuantizationConfig] = None


@json_schema_type
class BatchCompletionResponse(BaseModel):
    completion_message_batch: List[CompletionMessage]


@json_schema_type
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    sampling_params: Optional[SamplingParams] = SamplingParams()

    # zero-shot tool definitions as input to the model
    available_tools: Optional[List[ToolDefinition]] = Field(default_factory=list)

    stream: Optional[bool] = False
    logprobs: Optional[LogProbConfig] = None
    quantization_config: Optional[QuantizationConfig] = None


@json_schema_type
class ChatCompletionResponseStreamChunk(BaseModel):
    """SSE-stream of these events."""

    event: ChatCompletionResponseEvent


@json_schema_type
class ChatCompletionResponse(BaseModel):
    completion_message: CompletionMessage
    logprobs: Optional[List[TokenLogProbs]] = None


@json_schema_type
class BatchChatCompletionRequest(BaseModel):
    model: str
    messages_batch: List[List[Message]]
    sampling_params: Optional[SamplingParams] = SamplingParams()

    # zero-shot tool definitions as input to the model
    available_tools: Optional[List[ToolDefinition]] = Field(default_factory=list)

    logprobs: Optional[LogProbConfig] = None
    quantization_config: Optional[QuantizationConfig] = None


@json_schema_type
class BatchChatCompletionResponse(BaseModel):
    completion_message_batch: List[CompletionMessage]


class Inference(Protocol):

    @webmethod(route="/inference/completion")
    async def completion(
        self,
        request: CompletionRequest,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]: ...

    @webmethod(route="/inference/chat_completion")
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]: ...

    @webmethod(route="/inference/batch_completion")
    async def batch_completion(
        self,
        request: BatchCompletionRequest,
    ) -> BatchCompletionResponse: ...

    @webmethod(route="/inference/batch_chat_completion")
    async def batch_chat_completion(
        self,
        request: BatchChatCompletionRequest,
    ) -> BatchChatCompletionResponse: ...
