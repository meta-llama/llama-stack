# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import logging
from typing import AsyncGenerator, List, Optional, Union

import httpx
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk as OpenAIChatCompletionChunk,
)

from llama_stack.apis.common.content_types import (
    InterleavedContent,
    InterleavedContentItem,
    TextDelta,
    ToolCallDelta,
    ToolCallParseStatus,
)
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
    EmbeddingTaskType,
    Inference,
    LogProbConfig,
    Message,
    ResponseFormat,
    ResponseFormatType,
    SamplingParams,
    TextTruncation,
    ToolChoice,
    ToolConfig,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.apis.models import Model, ModelType
from llama_stack.models.llama.datatypes import BuiltinTool, StopReason, ToolCall
from llama_stack.models.llama.sku_list import all_registered_models
from llama_stack.providers.datatypes import ModelsProtocolPrivate
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_hf_repo_model_entry,
)
from llama_stack.providers.utils.inference.openai_compat import (
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


def build_hf_repo_model_entries():
    return [
        build_hf_repo_model_entry(
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

    return [
        ToolCall(
            call_id=call.id,
            tool_name=call.function.name,
            arguments=json.loads(call.function.arguments),
            arguments_json=call.function.arguments,
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

        # The tool.tool_name can be a str or a BuiltinTool enum. If
        # it's the latter, convert to a string.
        tool_name = tool.tool_name
        if isinstance(tool_name, BuiltinTool):
            tool_name = tool_name.value

        compat_tool = {
            "type": "function",
            "function": {
                "name": tool_name,
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
    stream: AsyncGenerator[OpenAIChatCompletionChunk, None],
) -> AsyncGenerator:
    event_type = ChatCompletionResponseEventType.start
    tool_call_buf = UnparseableToolCall()
    async for chunk in stream:
        if not chunk.choices:
            log.warning("vLLM failed to generation any completions - check the vLLM server logs for an error.")
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            args_str = tool_call_buf.arguments
            args = None
            try:
                args = {} if not args_str else json.loads(args_str)
            except Exception as e:
                log.warning(f"Failed to parse tool call buffer arguments: {args_str} \nError: {e}")
            if args:
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=event_type,
                        delta=ToolCallDelta(
                            tool_call=ToolCall(
                                call_id=tool_call_buf.call_id,
                                tool_name=tool_call_buf.tool_name,
                                arguments=args,
                                arguments_json=args_str,
                            ),
                            parse_status=ToolCallParseStatus.succeeded,
                        ),
                    )
                )
            elif args_str:
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
        self.register_helper = ModelRegistryHelper(build_hf_repo_model_entries())
        self.config = config
        self.client = None

    async def initialize(self) -> None:
        log.info(f"Initializing VLLM client with base_url={self.config.url}")
        self.client = AsyncOpenAI(
            base_url=self.config.url,
            api_key=self.config.api_token,
            http_client=None if self.config.tls_verify else httpx.AsyncClient(verify=False),
        )

    async def shutdown(self) -> None:
        pass

    async def unregister_model(self, model_id: str) -> None:
        pass

    async def completion(
        self,
        model_id: str,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]:
        if sampling_params is None:
            sampling_params = SamplingParams()
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
        sampling_params: Optional[SamplingParams] = None,
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        if sampling_params is None:
            sampling_params = SamplingParams()
        model = await self.model_store.get_model(model_id)
        # This is to be consistent with OpenAI API and support vLLM <= v0.6.3
        # References:
        #   * https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
        #   * https://github.com/vllm-project/vllm/pull/10000
        if not tools and tool_config is not None:
            tool_config.tool_choice = ToolChoice.none
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
        self, request: ChatCompletionRequest, client: AsyncOpenAI
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = await client.chat.completions.create(**params)
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

    async def _stream_chat_completion(self, request: ChatCompletionRequest, client: AsyncOpenAI) -> AsyncGenerator:
        params = await self._get_params(request)

        stream = await client.chat.completions.create(**params)
        if len(request.tools) > 0:
            res = _process_vllm_chat_completion_stream_response(stream)
        else:
            res = process_chat_completion_stream_response(stream, request)
        async for chunk in res:
            yield chunk

    async def _nonstream_completion(self, request: CompletionRequest) -> CompletionResponse:
        params = await self._get_params(request)
        r = await self.client.completions.create(**params)
        return process_completion_response(r)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        stream = await self.client.completions.create(**params)
        async for chunk in process_completion_stream_response(stream):
            yield chunk

    async def register_model(self, model: Model) -> Model:
        model = await self.register_helper.register_model(model)
        res = await self.client.models.list()
        available_models = [m.id async for m in res]
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
        contents: List[str] | List[InterleavedContentItem],
        text_truncation: Optional[TextTruncation] = TextTruncation.none,
        output_dimension: Optional[int] = None,
        task_type: Optional[EmbeddingTaskType] = None,
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)

        kwargs = {}
        assert model.model_type == ModelType.embedding
        assert model.metadata.get("embedding_dimension")
        kwargs["dimensions"] = model.metadata.get("embedding_dimension")
        assert all(not content_has_media(content) for content in contents), "VLLM does not support media for embeddings"
        response = await self.client.embeddings.create(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
            **kwargs,
        )

        embeddings = [data.embedding for data in response.data]
        return EmbeddingsResponse(embeddings=embeddings)
