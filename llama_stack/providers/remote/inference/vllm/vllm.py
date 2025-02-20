# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import logging
from typing import AsyncGenerator, List, Optional, Union

from llama_models.datatypes import StopReason, ToolCall
from openai import OpenAI

from llama_stack.apis.common.content_types import InterleavedContent, TextDelta, ToolCallDelta, ToolCallParseStatus
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_alias,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionResponse,
    UnparseableToolCall,
    convert_message_to_openai_dict,
    convert_tool_call,
    get_sampling_options,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    completion_request_to_prompt,
    content_has_media,
    interleaved_content_as_str,
    request_has_media,
)

from .config import VLLMInferenceAdapterConfig

log = logging.getLogger(__name__)


def build_hf_repo_model_aliases():
    return [
        build_hf_repo_model_alias(
            model.huggingface_repo,
            model.descriptor(),
        )
        for model in all_registered_models()
        if model.huggingface_repo
    ]


def _convert_to_vllm_tool_calls_in_response(
    tool_calls,
) -> List[ToolCall]:
    if not tool_calls:
        return []

    call_function_arguments = None
    for call in tool_calls:
        call_function_arguments = json.loads(call.function.arguments)

    return [
        ToolCall(
            call_id=call.id,
            tool_name=call.function.name,
            arguments=call_function_arguments,
        )
        for call in tool_calls
    ]


def _convert_to_vllm_tools_in_request(tools: List[ToolDefinition]) -> List[dict]:
    if tools is None:
        return tools

    compat_tools = []

    for tool in tools:
        properties = {}
        compat_required = []
        if tool.parameters:
            for tool_key, tool_param in tool.parameters.items():
                properties[tool_key] = {"type": tool_param.param_type}
                if tool_param.description:
                    properties[tool_key]["description"] = tool_param.description
                if tool_param.default:
                    properties[tool_key]["default"] = tool_param.default
                if tool_param.required:
                    compat_required.append(tool_key)

        compat_tool = {
            "type": "function",
            "function": {
                "name": tool.tool_name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": compat_required,
                },
            },
        }

        compat_tools.append(compat_tool)

    if len(compat_tools) > 0:
        return compat_tools
    return None


def _convert_to_vllm_finish_reason(finish_reason: str) -> StopReason:
    return {
        "stop": StopReason.end_of_turn,
        "length": StopReason.out_of_tokens,
        "tool_calls": StopReason.end_of_message,
    }.get(finish_reason, StopReason.end_of_turn)


async def _process_vllm_chat_completion_stream_response(
    stream: AsyncGenerator[OpenAICompatCompletionResponse, None],
) -> AsyncGenerator:
    event_type = ChatCompletionResponseEventType.start
    tool_call_buf = UnparseableToolCall()
    async for chunk in stream:
        choice = chunk.choices[0]
        if choice.finish_reason:
            args_str = tool_call_buf.arguments
            args = None
            try:
                args = {} if not args_str else json.loads(args_str)
            except Exception as e:
                log.warning(f"Failed to parse tool call buffer arguments: {args_str} \nError: {e}")
            if args is not None:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=event_type,
                        delta=ToolCallDelta(
                            tool_call=ToolCall(
                                call_id=tool_call_buf.call_id,
                                tool_name=tool_call_buf.tool_name,
                                arguments=args,
                            ),
                            parse_status=ToolCallParseStatus.succeeded,
                        ),
                    )
                )
            else:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.progress,
                        delta=ToolCallDelta(
                            tool_call=str(tool_call_buf),
                            parse_status=ToolCallParseStatus.failed,
                        ),
                    )
                )
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=ChatCompletionResponseEventType.complete,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=None,
                    stop_reason=_convert_to_vllm_finish_reason(choice.finish_reason),
                )
            )
        elif choice.delta.tool_calls:
            tool_call = convert_tool_call(choice.delta.tool_calls[0])
            tool_call_buf.tool_name += tool_call.tool_name
            tool_call_buf.call_id += tool_call.call_id
            tool_call_buf.arguments += tool_call.arguments
        else:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type=event_type,
                    delta=TextDelta(text=choice.delta.content or ""),
                    logprobs=None,
                )
            )
            event_type = ChatCompletionResponseEventType.progress


class VLLMInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, config: VLLMInferenceAdapterConfig) -> None:
        self.register_helper = ModelRegistryHelper(build_hf_repo_model_aliases())
        self.config = config
        self.client = None

    async def initialize(self) -> None:
        log.info(f"Initializing VLLM client with base_url={self.config.url}")
        self.client = OpenAI(base_url=self.config.url, api_key=self.config.api_token)

    async def shutdown(self) -> None:
        pass

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
        model = await self.model_store.get_model(model_id)
        request = CompletionRequest(
            model=model.provider_resource_id,
            content=content,
            sampling_params=sampling_params,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            stream=stream,
            logprobs=logprobs,
            response_format=response_format,
            tool_config=tool_config,
        )
        if stream:
            return self._stream_chat_completion(request, self.client)
        else:
            return await self._nonstream_chat_completion(request, self.client)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest, client: OpenAI
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = client.chat.completions.create(**params)
        choice = r.choices[0]
        result = ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=choice.message.content or "",
                stop_reason=_convert_to_vllm_finish_reason(choice.finish_reason),
                tool_calls=_convert_to_vllm_tool_calls_in_response(choice.message.tool_calls),
            ),
            logprobs=None,
        )
        return result

    async def _stream_chat_completion(self, request: ChatCompletionRequest, client: OpenAI) -> AsyncGenerator:
        params = await self._get_params(request)

        # TODO: Can we use client.completions.acreate() or maybe there is another way to directly create an async
        #  generator so this wrapper is not necessary?
        async def _to_async_generator():
            s = client.chat.completions.create(**params)
            for chunk in s:
                yield chunk

        stream = _to_async_generator()
        if len(request.tools) > 0:
            res = _process_vllm_chat_completion_stream_response(stream)
        else:
            res = process_chat_completion_stream_response(stream, request)
        async for chunk in res:
            yield chunk

    async def _nonstream_completion(self, request: CompletionRequest) -> CompletionResponse:
        params = await self._get_params(request)
        r = self.client.completions.create(**params)
        return process_completion_response(r)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        # Wrapper for async generator similar
        async def _to_async_generator():
            stream = self.client.completions.create(**params)
            for chunk in stream:
                yield chunk

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def register_model(self, model: Model) -> Model:
        model = await self.register_helper.register_model(model)
        res = self.client.models.list()
        available_models = [m.id for m in res]
        if model.provider_resource_id not in available_models:
            raise ValueError(
                f"Model {model.provider_resource_id} is not being served by vLLM. "
                f"Available models: {', '.join(available_models)}"
            )
        return model

    async def _get_params(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> dict:
        options = get_sampling_options(request.sampling_params)
        if "max_tokens" not in options:
            options["max_tokens"] = self.config.max_tokens

        input_dict = {}
        if isinstance(request, ChatCompletionRequest) and request.tools is not None:
            input_dict = {"tools": _convert_to_vllm_tools_in_request(request.tools)}

        if isinstance(request, ChatCompletionRequest):
            input_dict["messages"] = [await convert_message_to_openai_dict(m, download=True) for m in request.messages]
        else:
            assert not request_has_media(request), "vLLM does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(request)

        if fmt := request.response_format:
            if fmt.type == ResponseFormatType.json_schema.value:
                input_dict["extra_body"] = {"guided_json": request.response_format.json_schema}
            elif fmt.type == ResponseFormatType.grammar.value:
                raise NotImplementedError("Grammar response format not supported yet")
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        if request.logprobs and request.logprobs.top_k:
            input_dict["logprobs"] = request.logprobs.top_k

        return {
            "model": request.model,
            **input_dict,
            "stream": request.stream,
            **options,
        }

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)

        kwargs = {}
        assert model.model_type == ModelType.embedding
        assert model.metadata.get("embedding_dimensions")
        kwargs["dimensions"] = model.metadata.get("embedding_dimensions")
        assert all(not content_has_media(content) for content in contents), "VLLM does not support media for embeddings"
        response = self.client.embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
            **kwargs,
        )

        embeddings = [data.embedding for data in response.data]
        return EmbeddingsResponse(embeddings=embeddings)
