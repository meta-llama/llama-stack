import asyncio
from typing import AsyncIterator, AsyncGenerator, List, Literal, Optional, Union
import lmstudio as lms

from llama_stack.apis.common.content_types import InterleavedContent, TextDelta
from llama_stack.apis.inference.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionResponse,
    CompletionResponseStreamChunk,
    JsonSchemaResponseFormat,
    Message,
    ToolConfig,
    ToolDefinition,
)
from llama_stack.models.llama.datatypes import (
    GreedySamplingStrategy,
    SamplingParams,
    StopReason,
    TopKSamplingStrategy,
    TopPSamplingStrategy,
)
from llama_stack.providers.utils.inference.openai_compat import (
    convert_message_to_openai_dict_new,
    convert_openai_chat_completion_choice,
    convert_openai_chat_completion_stream,
    convert_tooldef_to_openai_tool,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    content_has_media,
    interleaved_content_as_str,
)
from openai import AsyncOpenAI as OpenAI

LlmPredictionStopReason = Literal[
    "userStopped",
    "modelUnloaded",
    "failed",
    "eosFound",
    "stopStringFound",
    "toolCalls",
    "maxPredictedTokensReached",
    "contextLengthReached",
]


class LMStudioClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.sdk_client = lms.Client(self.url)
        self.openai_client = OpenAI(base_url=f"http://{url}/v1", api_key="lmstudio")

    async def check_if_model_present_in_lmstudio(self, provider_model_id):
        models = await asyncio.to_thread(self.sdk_client.list_downloaded_models)
        model_ids = [m.model_key for m in models]
        if provider_model_id in model_ids:
            return True

        model_ids = [id.split("/")[-1] for id in model_ids]
        if provider_model_id in model_ids:
            return True
        return False

    async def get_embedding_model(self, provider_model_id: str):
        model = await asyncio.to_thread(
            self.sdk_client.embedding.model, provider_model_id
        )
        return model

    async def embed(
        self, embedding_model: lms.EmbeddingModel, contents: Union[str, List[str]]
    ):
        embeddings = await asyncio.to_thread(embedding_model.embed, contents)
        return embeddings

    async def get_llm(self, provider_model_id: str) -> lms.LLM:
        model = await asyncio.to_thread(self.sdk_client.llm.model, provider_model_id)
        return model

    async def _llm_respond_non_tools(
        self,
        llm: lms.LLM,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        json_schema: Optional[JsonSchemaResponseFormat] = None,
        stream: Optional[bool] = False,
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:
        chat = self._convert_message_list_to_lmstudio_chat(messages)
        config = self._get_completion_config_from_params(sampling_params)
        if stream:

            async def stream_generator():
                prediction_stream = await asyncio.to_thread(
                    llm.respond_stream,
                    history=chat,
                    config=config,
                    response_format=json_schema,
                )

                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.start,
                        delta=TextDelta(text=""),
                    )
                )
                async for chunk in self._async_iterate(prediction_stream):
                    yield ChatCompletionResponseStreamChunk(
                        event=ChatCompletionResponseEvent(
                            event_type=ChatCompletionResponseEventType.progress,
                            delta=TextDelta(text=chunk.content),
                        )
                    )
                yield ChatCompletionResponseStreamChunk(
                    event=ChatCompletionResponseEvent(
                        event_type=ChatCompletionResponseEventType.complete,
                        delta=TextDelta(text=""),
                    )
                )

            return stream_generator()
        else:
            response = await asyncio.to_thread(
                llm.respond,
                history=chat,
                config=config,
                response_format=json_schema,
            )
            return self._convert_prediction_to_chat_response(response)

    async def _llm_respond_with_tools(
        self,
        llm: lms.LLM,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        json_schema: Optional[JsonSchemaResponseFormat] = None,
        stream: Optional[bool] = False,
        tools: Optional[List[ToolDefinition]] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:
        model_key = llm.get_info().model_key
        request = ChatCompletionRequest(
            model=model_key,
            messages=messages,
            sampling_params=sampling_params,
            response_format=json_schema,
            tools=tools,
            tool_config=tool_config,
            stream=stream,
        )
        rest_request = await self._convert_request_to_rest_call(request)
        if stream:
            stream = await self.openai_client.chat.completions.create(**rest_request)
            return convert_openai_chat_completion_stream(
                stream, enable_incremental_tool_calls=True
            )
        response = await self.openai_client.chat.completions.create(**rest_request)
        if response:
            result = convert_openai_chat_completion_choice(response.choices[0])
            return result
        else:
            return None

    async def llm_respond(
        self,
        llm: lms.LLM,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        json_schema: Optional[JsonSchemaResponseFormat] = None,
        stream: Optional[bool] = False,
        tools: Optional[List[ToolDefinition]] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:
        if tools is None or len(tools) == 0:
            return await self._llm_respond_non_tools(
                llm=llm,
                messages=messages,
                sampling_params=sampling_params,
                json_schema=json_schema,
                stream=stream,
            )
        else:
            return await self._llm_respond_with_tools(
                llm=llm,
                messages=messages,
                sampling_params=sampling_params,
                json_schema=json_schema,
                stream=stream,
                tools=tools,
                tool_config=tool_config,
            )

    async def llm_completion(
        self,
        llm: lms.LLM,
        content: InterleavedContent,
        sampling_params: Optional[SamplingParams] = None,
        json_schema: Optional[JsonSchemaResponseFormat] = None,
        stream: Optional[bool] = False,
    ) -> Union[CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]]:
        config = self._get_completion_config_from_params(sampling_params)
        if stream:

            async def stream_generator():
                prediction_stream = await asyncio.to_thread(
                    llm.complete_stream,
                    prompt=interleaved_content_as_str(content),
                    config=config,
                    response_format=json_schema,
                )
                async for chunk in self._async_iterate(prediction_stream):
                    yield CompletionResponseStreamChunk(
                        delta=chunk.content,
                    )

            return stream_generator()
        else:
            response = await asyncio.to_thread(
                llm.complete,
                prompt=interleaved_content_as_str(content),
                config=config,
                response_format=json_schema,
            )
            return CompletionResponse(
                content=response.content,
                stop_reason=self._get_stop_reason(response.stats.stop_reason),
            )

    def _convert_message_list_to_lmstudio_chat(
        self, messages: List[Message]
    ) -> lms.Chat:
        chat = lms.Chat()
        for message in messages:
            if content_has_media(message.content):
                raise NotImplementedError(
                    "Media content is not supported in LMStudio messages"
                )
            if message.role == "user":
                chat.add_user_message(interleaved_content_as_str(message.content))
            elif message.role == "system":
                chat.add_system_prompt(interleaved_content_as_str(message.content))
            elif message.role == "assistant":
                chat.add_assistant_response(interleaved_content_as_str(message.content))
            else:
                raise ValueError(f"Unsupported message role: {message.role}")
        return chat

    def _convert_prediction_to_chat_response(
        self, result: lms.PredictionResult
    ) -> ChatCompletionResponse:
        response = ChatCompletionResponse(
            completion_message=CompletionMessage(
                content=result.content,
                stop_reason=self._get_stop_reason(result.stats.stop_reason),
                tool_calls=None,
            )
        )
        return response

    def _get_completion_config_from_params(
        self,
        params: Optional[SamplingParams] = None,
    ) -> lms.LlmPredictionConfigDict:
        options = lms.LlmPredictionConfigDict()
        if params is None:
            return options
        if isinstance(params.strategy, GreedySamplingStrategy):
            options.update({"temperature": 0.0})
        elif isinstance(params.strategy, TopPSamplingStrategy):
            options.update(
                {
                    "temperature": params.strategy.temperature,
                    "top_p": params.strategy.top_p,
                }
            )
        elif isinstance(params.strategy, TopKSamplingStrategy):
            options.update({"topKSampling": params.strategy.top_k})
        else:
            raise ValueError(f"Unsupported sampling strategy: {params.strategy}")
        options.update(
            {
                "maxTokens": params.max_tokens if params.max_tokens != 0 else None,
                "repetitionPenalty": (
                    params.repetition_penalty
                    if params.repetition_penalty != 0
                    else None
                ),
            }
        )
        return options

    def _get_stop_reason(self, stop_reason: LlmPredictionStopReason) -> StopReason:
        if stop_reason == "eosFound":
            return StopReason.end_of_message
        elif stop_reason == "maxPredictedTokensReached":
            return StopReason.out_of_tokens
        else:
            return StopReason.end_of_turn

    async def _async_iterate(self, iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield await asyncio.to_thread(next, iterator)
            except:
                break

    async def _convert_request_to_rest_call(
        self, request: ChatCompletionRequest
    ) -> dict:
        compatible_request = self._convert_sampling_params(request.sampling_params)
        compatible_request["model"] = request.model
        compatible_request["messages"] = [
            await convert_message_to_openai_dict_new(m) for m in request.messages
        ]
        if request.response_format:
            compatible_request["response_format"] = {
                "type": "json_schema",
                "json_schema": request.response_format.json_schema,
            }
        if request.tools is not None:
            compatible_request["tools"] = [
                convert_tooldef_to_openai_tool(tool) for tool in request.tools
            ]
        compatible_request["logprobs"] = False
        compatible_request["stream"] = request.stream
        compatible_request["extra_headers"] = {
            b"User-Agent": b"llama-stack: lmstudio-inference-adapter"
        }
        return compatible_request

    def _convert_sampling_params(self, sampling_params: Optional[SamplingParams]) -> dict:
        params = {}

        if sampling_params is None:
            return params
        params["frequency_penalty"] = sampling_params.repetition_penalty

        if sampling_params.max_tokens:
            params["max_completion_tokens"] = sampling_params.max_tokens

        if isinstance(sampling_params.strategy, TopPSamplingStrategy):
            params["top_p"] = sampling_params.strategy.top_p
        if isinstance(sampling_params.strategy, TopKSamplingStrategy):
            params["extra_body"]["top_k"] = sampling_params.strategy.top_k
        if isinstance(sampling_params.strategy, GreedySamplingStrategy):
            params["temperature"] = 0.0

        return params
