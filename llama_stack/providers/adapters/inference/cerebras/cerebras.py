# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json

from typing import AsyncGenerator

from cerebras.cloud.sdk import Cerebras
from cerebras.cloud.sdk.types.chat.completion_create_params import (
    Message as CerebrasMessage,
    MessageAssistantMessageRequestToolCallFunctionTyped,
    MessageAssistantMessageRequestToolCallTyped,
    MessageAssistantMessageRequestTyped,
    MessageSystemMessageRequestTyped,
    MessageToolMessageRequestTyped,
    MessageUserMessageRequestTyped,
    Tool,
    ToolFunctionTyped,
    ToolTyped,
)

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.inference import *  # noqa: F403

from pydantic import BaseModel

from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)

from .config import CerebrasImplConfig


CEREBRAS_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "llama3.1-8b",
    "Llama3.1-70B-Instruct": "llama3.1-70b",
}


class CerebrasInferenceAdapter(ModelRegistryHelper, Inference):
    def __init__(self, config: CerebrasImplConfig) -> None:
        ModelRegistryHelper.__init__(
            self, stack_to_provider_models_map=CEREBRAS_SUPPORTED_MODELS
        )
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

        self.client = Cerebras(
            base_url=self.config.base_url, api_key=self.config.api_key
        )

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()

    def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]:
        request = ChatCompletionRequest(
            model=model,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )

        if stream:
            return self._stream_chat_completion(request, self.client)
        else:
            return self._nonstream_chat_completion(request, self.client)

    def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: Cerebras
    ) -> ChatCompletionResponse:
        params = self._get_params(request)

        r = client.chat.completions.create(**params)
        return process_chat_completion_response(request, r, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest, client: Cerebras
    ) -> AsyncGenerator:
        params = self._get_params(request)

        async def _to_async_generator():
            s = client.chat.completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(
            request, stream, self.formatter
        ):
            yield chunk

    def _get_params(self, request: ChatCompletionRequest) -> dict:
        if request.sampling_params and request.sampling_params.top_k:
            raise ValueError("`top_k` not supported by Cerebras")

        return {
            "model": self.map_to_provider_model(request.model),
            "messages": self._construct_cerebras_messages(request),
            "tools": self._construct_cerebras_tools(request),
            "tool_choice": request.tool_choice.value if request.tool_choice else None,
            "stream": request.stream,
            "logprobs": request.logprobs is not None,
            "top_logprobs": request.logprobs,
            **get_sampling_options(request),
        }

    @staticmethod
    def _construct_cerebras_tools(request: ChatCompletionRequest) -> List[Tool]:
        tools = []

        for raw_tool in request.tools:
            tools.append(
                ToolTyped(
                    function=ToolFunctionTyped(
                        name=__class__._parse_tool_name(raw_tool.tool_name),
                        description=raw_tool.description,
                        parameters=(
                            {
                                k: v.model_dump() if isinstance(v, BaseModel) else v
                                for k, v in raw_tool.parameters.items()
                            }
                            if raw_tool.parameters
                            else None
                        ),
                    ),
                    type="object",
                )
            )

        return tools

    @staticmethod
    def _construct_cerebras_messages(
        request: ChatCompletionRequest,
    ) -> List[CerebrasMessage]:
        messages = []

        for raw_message in request.messages:
            content = raw_message.content

            if not isinstance(content, str):
                raise ValueError(
                    f"Message content must be of type `str` but got `{type(content)}`"
                )

            if isinstance(raw_message, UserMessage):
                messages.append(
                    MessageUserMessageRequestTyped(
                        content=content,
                        role="user",
                    )
                )
            elif isinstance(raw_message, SystemMessage):
                messages.append(
                    MessageSystemMessageRequestTyped(
                        content=content,
                        role="system",
                    )
                )
            elif isinstance(raw_message, ToolResponseMessage):
                messages.append(
                    MessageToolMessageRequestTyped(
                        role="tool",
                        tool_call_id=raw_message.call_id,
                        name=__class__._parse_tool_name(raw_message.tool_name),
                        content=content,
                    )
                )
            elif isinstance(raw_message, CompletionMessage):
                messages.append(
                    MessageAssistantMessageRequestTyped(
                        role="assistant",
                        content=content,
                        tool_calls=__class__._construct_cerebras_tool_calls(
                            raw_message.tool_calls
                        ),
                    )
                )

        return messages

    @staticmethod
    def _construct_cerebras_tool_calls(
        raw_tool_calls: List[ToolCall],
    ) -> List[MessageAssistantMessageRequestToolCallTyped]:
        return [
            MessageAssistantMessageRequestToolCallTyped(
                id=tool_call.call_id,
                type="function",
                function=MessageAssistantMessageRequestToolCallFunctionTyped(
                    arguments=json.dumps(tool_call.arguments),
                    # Handle BuiltinTool using enum name names.
                    name=__class__._parse_tool_name(tool_call.tool_name),
                ),
            )
            for tool_call in raw_tool_calls
        ]

    @staticmethod
    def _parse_tool_name(raw_tool_name: Union[str, Enum]) -> str:
        return raw_tool_name if isinstance(raw_tool_name, str) else raw_tool_name.value

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
