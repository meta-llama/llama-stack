from typing import AsyncGenerator

from openai import OpenAI
import json
from llama_models.datatypes import CoreModelId

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    ModelRegistryHelper,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    request_has_media,
)
from .config import GroqImplConfig

MODEL_ALIASES = [
    build_model_alias(
        "llama-3.1-8b-instant",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "llama-3.1-70b-versatile",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "llama-3.2-1b-preview",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_alias(
        "llama-3.2-3b-preview",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "llama-3.2-11b-vision-preview",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "llama-3.2-90b-vision-preview",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_model_alias(
        "llama-guard-3-8b",
        CoreModelId.llama_guard_3_8b.value,
    ),
]


class GroqInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):
    def __init__(self, config: GroqImplConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ALIASES)
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def _get_client(self) -> OpenAI:
        groq_api_key = None
        if self.config.api_key is not None:
            groq_api_key = self.config.api_key
        else:
            provider_data = self.get_request_provider_data()
            if provider_data is None or not provider_data.groq_api_key:
                raise ValueError(
                    'Pass Groq API Key in the header X-LlamaStack-ProviderData as { "groq_api_key": <your api key> }'
                )
            groq_api_key = provider_data.groq_api_key
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=groq_api_key)


    async def completion(
        self,
        model_id: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError(
            "Groq does not support text completion. See chat completion in the documentation instead: https://console.groq.com/docs/api-reference#chat-create"
        )

    async def chat_completion(
        self,
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        model = await self.model_store.get_model(model_id)
        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            response_format=response_format,
            stream=stream,
            logprobs=logprobs,
        )

        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)
        r = self._get_client().chat.completions.create(**params)
        return self._process_chat_completion_response(r)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator[ChatCompletionResponseStreamChunk, None]:
        params = await self._get_params(request)
        
        raw_stream = self._get_client().chat.completions.create(**params)
        
        async for stream_chunk in self._process_chat_completion_stream_response(raw_stream):
            yield stream_chunk


    async def _get_params(
        self, request: ChatCompletionRequest
    ) -> dict:
        params = {
            "model": request.model,
            "stream": request.stream,
        }

        # Process messages
        params["messages"] = [
            {
                "role": m.role,
                "content": m.content,
            }
            for m in request.messages
        ]

        # Build options
        options = self._build_options(
            request.sampling_params, request.response_format, request.logprobs
        )
        params.update(options)

        # Handle tools and tool_choice
        if request.tools:
            params["tools"] = []
            for tool in request.tools:
                # Convert the ToolDefinition into the desired format
                params["tools"].append({
                    "type": "function",
                    "function": {
                        "name": str(tool.tool_name.value if hasattr(tool.tool_name, 'value') else tool.tool_name),
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                param_name: {
                                    "type": param.param_type,
                                    "description": param.description,
                                }
                                for param_name, param in tool.parameters.items()
                            },
                            "required": [
                                param_name
                                for param_name, param in tool.parameters.items()
                                if param.required
                            ],
                        },
                    },
                })

        if request.tool_choice:
            params["tool_choice"] = request.tool_choice.value

        return params

    def _build_options(
        self,
        sampling_params: Optional[SamplingParams],
        fmt: Optional[ResponseFormat],
        logprobs: Optional[LogProbConfig],
    ) -> dict:
        options = {}
        if sampling_params:
            if sampling_params.temperature is not None:
                options["temperature"] = sampling_params.temperature
            if sampling_params.max_tokens and sampling_params.max_tokens > 0:
                options["max_tokens"] = sampling_params.max_tokens
            if sampling_params.top_p is not None:
                options["top_p"] = sampling_params.top_p
            # The following parameters are not supported by Groq API
            # if sampling_params.top_k is not None:
            #     options["top_k"] = sampling_params.top_k
            # if sampling_params.repetition_penalty is not None:
            #     options["repetition_penalty"] = sampling_params.repetition_penalty

        if fmt:
            if fmt.type == ResponseFormatType.json_schema.value:
                options["response_format"] = {
                    "type": "json_object",
                    "schema": fmt.json_schema,
                }
            else:
                raise ValueError(f"Unknown response format {fmt.type}")

        if logprobs:
            if logprobs.top_k is not None and logprobs.top_k > 0:
                options["logprobs"] = True
                options["top_logprobs"] = logprobs.top_k
            else:
                options["logprobs"] = False

        return options


    def _process_chat_completion_response(self, response):
        # Ensure response is an object with a `choices` attribute
        if not hasattr(response, 'choices') or not isinstance(response.choices, list):
            raise ValueError("Invalid response format: 'choices' attribute is missing or not a list.")
        
        first_choice = response.choices[0]
        
        # Ensure the first choice has a valid `message` field
        if not hasattr(first_choice, 'message') or not first_choice.message:
            raise ValueError("Invalid response format: 'message' field is missing in the first choice.")
        
        tool_calls = []
        for tool_call in (first_choice.message.tool_calls or []):
            arguments = getattr(tool_call.function, 'arguments', {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            
            # Append transformed ToolCall
            tool_calls.append(ToolCall(
                call_id=getattr(tool_call, 'id', 'unknown_call_id'),
                tool_name=getattr(tool_call.function, 'name', 'unknown_tool'),
                arguments=arguments
            ))
        
        content = first_choice.message.content
        if content is None:
            content = ""  # Provide a default empty string
        
        finish_reason = {
            "stop": StopReason.end_of_turn,
            "length": StopReason.out_of_tokens,
            "tool_calls": StopReason.end_of_message,
        }.get(getattr(first_choice, 'finish_reason', None), StopReason.end_of_turn)

        completion_message = CompletionMessage(
            role=first_choice.message.role,
            content=content,
            stop_reason=finish_reason,
            tool_calls=tool_calls,
        )
        
        return ChatCompletionResponse(
            completion_message=completion_message,
            logprobs=None  # Groq does not provide logprobs currently. See reference for latest: https://console.groq.com/docs/api-reference#chat-create
        )


    def _convert_chunk_to_stream_chunk(self, chunk):
        if not chunk.choices or len(chunk.choices) == 0:
            return None
    
        choice = chunk.choices[0]
        delta = choice.delta

        # Handle tool calls in full form directly
        tool_calls = []
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                arguments = tool_call.function.arguments
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                # Append transformed ToolCall
                tool_calls.append(ToolCall(
                    call_id=tool_call.id,
                    tool_name=tool_call.function.name,
                    arguments=arguments
                ))

        # Determine event type
        if choice.finish_reason == 'stop' or choice.finish_reason == 'tool_calls':
            event_type = ChatCompletionResponseEventType.complete
        elif delta and delta.role == 'assistant' and not delta.content:
            event_type = ChatCompletionResponseEventType.start
        else:
            event_type = ChatCompletionResponseEventType.progress
    
        # Handle delta content
        if delta.content is not None:
            event_delta = delta.content
        elif tool_calls:
            # Construct ToolCallDelta if tool calls exist
            event_delta = ToolCallDelta(
                content=tool_calls[0], # Tools currently come once per chunk, and thus, we can sample the first tool as there will not be more than one here.
                parse_status=ToolCallParseStatus("success") # Groq currently only returns tool calls in one chunk. If a tool call is there, it is complete and has success status.
            )
        elif choice.finish_reason == 'stop':
            # For 'stop' events with no content, set delta to empty string
            event_delta = ""
        else:
            # For non-stop events with no content, set delta to empty string
            event_delta = ""

        finish_reason = {
            "stop": StopReason.end_of_turn,
            "length": StopReason.out_of_tokens,
            "tool_calls": StopReason.end_of_message,
        }.get(choice.finish_reason, StopReason.end_of_turn)

        # Construct the event
        event = ChatCompletionResponseEvent(
            event_type=event_type,
            delta=event_delta,
            stop_reason=finish_reason,
            logprobs=choice.logprobs,
        )

        # Create the stream chunk
        stream_chunk = ChatCompletionResponseStreamChunk(event=event)
        return stream_chunk


    async def _process_chat_completion_stream_response(self, stream):
        if hasattr(stream, "__aiter__"):
            # Consume as an async iterable
            async for chunk in stream:
                stream_chunk = self._convert_chunk_to_stream_chunk(chunk)
                if stream_chunk:
                    yield stream_chunk
        elif hasattr(stream, "__iter__"):
            # Wrap sync iterable in an async generator
            for chunk in stream:
                stream_chunk = self._convert_chunk_to_stream_chunk(chunk)
                if stream_chunk:
                    yield stream_chunk
        else:
            raise TypeError(f"'stream' object is not iterable: {type(stream)}")


    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
