# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import AsyncGenerator, List, Optional

from openai import OpenAI

from llama_stack.apis.common.content_types import (
    ImageContentItem,
    InterleavedContent,
    InterleavedContentItem,
    TextContentItem,
)
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionMessage,
    EmbeddingsResponse,
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    StopReason,
    SystemMessage,
    TextTruncation,
    ToolCall,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.models.llama.datatypes import (
    GreedySamplingStrategy,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.providers.utils.inference.model_registry import ModelRegistryHelper
from llama_stack.providers.utils.inference.openai_compat import (
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    convert_image_content_to_url,
)

from .config import SambaNovaImplConfig
from .models import MODEL_ENTRIES


class SambaNovaInferenceAdapter(ModelRegistryHelper, Inference):
    def __init__(self, config: SambaNovaImplConfig) -> None:
        ModelRegistryHelper.__init__(self, model_entries=MODEL_ENTRIES)
        self.config = config

    async def initialize(self) -> None:
        return

    async def shutdown(self) -> None:
        pass

    def _get_client(self) -> OpenAI:
        return OpenAI(base_url=self.config.url, api_key=self.config.api_key)

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        tool_config: Optional[ToolConfig] = None,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)

        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            stream=stream,
            logprobs=logprobs,
            tool_config=tool_config,
        )
        request_sambanova = await self.convert_chat_completion_request(request)

        if stream:
            return self._stream_chat_completion(request_sambanova)
        else:
            return await self._nonstream_chat_completion(request_sambanova)

    async def _nonstream_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        response = self._get_client().chat.completions.create(**request)

        choice = response.choices[0]

        result = ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=choice.message.content or "",
                stop_reason=self.convert_to_sambanova_finish_reason(choice.finish_reason),
                tool_calls=self.convert_to_sambanova_tool_calls(choice.message.tool_calls),
            ),
            logprobs=None,
        )

        return result

    async def _stream_chat_completion(self, request: ChatCompletionRequest) -> AsyncGenerator:
        async def _to_async_generator():
            streaming = self._get_client().chat.completions.create(**request)
            for chunk in streaming:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(stream, request):
            yield chunk

    async def embeddings(
        self,
        model_id: str,
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        raise NotImplementedError()

    async def convert_chat_completion_request(self, request: ChatCompletionRequest) -> dict:
        compatible_request = self.convert_sampling_params(request.sampling_params)
        compatible_request["model"] = request.model
        compatible_request["messages"] = await self.convert_to_sambanova_messages(request.messages)
        compatible_request["stream"] = request.stream
        compatible_request["logprobs"] = False
        compatible_request["extra_headers"] = {
            b"User-Agent": b"llama-stack: sambanova-inference-adapter",
        }
        compatible_request["tools"] = self.convert_to_sambanova_tool(request.tools)
        return compatible_request

    def convert_sampling_params(self, sampling_params: SamplingParams, legacy: bool = False) -> dict:
        params = {}

        if sampling_params:
            params["frequency_penalty"] = sampling_params.repetition_penalty

            if sampling_params.max_tokens:
                if legacy:
                    params["max_tokens"] = sampling_params.max_tokens
                else:
                    params["max_completion_tokens"] = sampling_params.max_tokens

            if isinstance(sampling_params.strategy, TopPSamplingStrategy):
                params["top_p"] = sampling_params.strategy.top_p
            if isinstance(sampling_params.strategy, TopKSamplingStrategy):
                params["extra_body"]["top_k"] = sampling_params.strategy.top_k
            if isinstance(sampling_params.strategy, GreedySamplingStrategy):
                params["temperature"] = 0.0

        return params

    async def convert_to_sambanova_messages(self, messages: List[Message]) -> List[dict]:
        conversation = []
        for message in messages:
            content = {}

            content["content"] = await self.convert_to_sambanova_content(message)

            if isinstance(message, UserMessage):
                content["role"] = "user"
            elif isinstance(message, CompletionMessage):
                content["role"] = "assistant"
                tools = []
                for tool_call in message.tool_calls:
                    tools.append(
                        {
                            "id": tool_call.call_id,
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments),
                            },
                            "type": "function",
                        }
                    )
                content["tool_calls"] = tools
            elif isinstance(message, ToolResponseMessage):
                content["role"] = "tool"
                content["tool_call_id"] = message.call_id
            elif isinstance(message, SystemMessage):
                content["role"] = "system"

            conversation.append(content)

        return conversation

    async def convert_to_sambanova_content(self, message: Message) -> dict:
        async def _convert_content(content) -> dict:
            if isinstance(content, ImageContentItem):
                url = await convert_image_content_to_url(content, download=True)
                # A fix to make sure the call sucess.
                components = url.split(";base64")
                url = f"{components[0].lower()};base64{components[1]}"
                return {
                    "type": "image_url",
                    "image_url": {"url": url},
                }
            else:
                text = content.text if isinstance(content, TextContentItem) else content
                assert isinstance(text, str)
                return {"type": "text", "text": text}

        if isinstance(message.content, list):
            # If it is a list, the text content should be wrapped in dict
            content = [await _convert_content(c) for c in message.content]
        else:
            content = message.content

        return content

    def convert_to_sambanova_tool(self, tools: List[ToolDefinition]) -> List[dict]:
        if tools is None:
            return tools

        compatiable_tools = []

        for tool in tools:
            properties = {}
            compatiable_required = []
            if tool.parameters:
                for tool_key, tool_param in tool.parameters.items():
                    properties[tool_key] = {"type": tool_param.param_type}
                    if tool_param.description:
                        properties[tool_key]["description"] = tool_param.description
                    if tool_param.default:
                        properties[tool_key]["default"] = tool_param.default
                    if tool_param.required:
                        compatiable_required.append(tool_key)

            compatiable_tool = {
                "type": "function",
                "function": {
                    "name": tool.tool_name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": compatiable_required,
                    },
                },
            }

            compatiable_tools.append(compatiable_tool)

        if len(compatiable_tools) > 0:
            return compatiable_tools
        return None

    def convert_to_sambanova_finish_reason(self, finish_reason: str) -> StopReason:
        return {
            "stop": StopReason.end_of_turn,
            "length": StopReason.out_of_tokens,
            "tool_calls": StopReason.end_of_message,
        }.get(finish_reason, StopReason.end_of_turn)

    def convert_to_sambanova_tool_calls(
        self,
        tool_calls,
    ) -> List[ToolCall]:
        if not tool_calls:
            return []

        compitable_tool_calls = [
            ToolCall(
                call_id=call.id,
                tool_name=call.function.name,
                arguments=json.loads(call.function.arguments),
                arguments_json=call.function.arguments,
            )
            for call in tool_calls
        ]

        return compitable_tool_calls
