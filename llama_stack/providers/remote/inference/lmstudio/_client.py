# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import logging
import re
from typing import Any, AsyncIterator, List, Literal, Optional, Union

import lmstudio as lms
from openai import AsyncOpenAI as OpenAI

from llama_stack.apis.common.content_types import InterleavedContent, TextDelta
from llama_stack.apis.inference import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseEventType,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    CompletionResponse,
    CompletionResponseStreamChunk,
    GrammarResponseFormat,
    GreedySamplingStrategy,
    JsonSchemaResponseFormat,
    Message,
    SamplingParams,
    StopReason,
    ToolConfig,
    ToolDefinition,
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

    # Standard error handling helper methods
    def _log_error(self, error, context=""):
        """Centralized error logging method"""
        logging.warning(f"Error in LMStudio {context}: {error}")

    async def _create_fallback_chat_stream(
        self, error_message="I encountered an error processing your request."
    ) -> AsyncIterator[ChatCompletionResponseStreamChunk]:
        """Create a standardized fallback stream for chat completions"""
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.start,
                delta=TextDelta(text=""),
            )
        )
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.progress,
                delta=TextDelta(text=error_message),
            )
        )
        yield ChatCompletionResponseStreamChunk(
            event=ChatCompletionResponseEvent(
                event_type=ChatCompletionResponseEventType.complete,
                delta=TextDelta(text=""),
            )
        )

    async def _create_fallback_completion_stream(self, error_message="Error processing response"):
        """Create a standardized fallback stream for text completions"""
        yield CompletionResponseStreamChunk(
            delta=error_message,
        )

    def _create_fallback_chat_response(
        self, error_message="I encountered an error processing your request."
    ) -> ChatCompletionResponse:
        """Create a standardized fallback response for chat completions"""
        return ChatCompletionResponse(
            completion_message=CompletionMessage(
                role="assistant",
                content=error_message,
                stop_reason=StopReason.end_of_message,
            )
        )

    def _create_fallback_completion_response(self, error_message="Error processing response") -> CompletionResponse:
        """Create a standardized fallback response for text completions"""
        return CompletionResponse(
            content=error_message,
            stop_reason=StopReason.end_of_message,
        )

    def _handle_json_extraction(self, content, context="JSON extraction"):
        """Standardized method to extract valid JSON from potentially malformed content"""
        try:
            json_content = json.loads(content)
            return json.dumps(json_content)  # Re-serialize to ensure valid JSON
        except json.JSONDecodeError as e:
            self._log_error(e, f"{context} - Attempting to extract valid JSON")

            json_patterns = [
                r"(\{.*\})",  # Match anything between curly braces
                r"(\[.*\])",  # Match anything between square brackets
                r"```json\s*([\s\S]*?)\s*```",  # Match content in JSON code blocks
                r"```\s*([\s\S]*?)\s*```",  # Match content in any code blocks
            ]

            for pattern in json_patterns:
                json_match = re.search(pattern, content, re.DOTALL)
                if json_match:
                    valid_json = json_match.group(1)
                    try:
                        json_content = json.loads(valid_json)
                        return json.dumps(json_content)  # Re-serialize to ensure valid JSON
                    except json.JSONDecodeError:
                        continue  # Try the next pattern

            # If we couldn't extract valid JSON, log a warning
            self._log_error("Failed to extract valid JSON", context)
            return None

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
        model = await asyncio.to_thread(self.sdk_client.embedding.model, provider_model_id)
        return model

    async def embed(self, embedding_model: lms.EmbeddingModel, contents: Union[str, List[str]]):
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
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        chat = self._convert_message_list_to_lmstudio_chat(messages)
        config = self._get_completion_config_from_params(sampling_params)
        if stream:

            async def stream_generator() -> AsyncIterator[ChatCompletionResponseStreamChunk]:
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
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        try:
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
                try:
                    stream = await self.openai_client.chat.completions.create(**rest_request)
                    return convert_openai_chat_completion_stream(stream, enable_incremental_tool_calls=True)
                except Exception as e:
                    self._log_error(e, "streaming tool calling")
                    return self._create_fallback_chat_stream()

            try:
                response = await self.openai_client.chat.completions.create(**rest_request)
                if response:
                    result = convert_openai_chat_completion_choice(response.choices[0])
                    return result
                else:
                    # Handle empty response
                    self._log_error("Empty response from OpenAI API", "chat completion")
                    return self._create_fallback_chat_response()
            except Exception as e:
                self._log_error(e, "non-streaming tool calling")
                return self._create_fallback_chat_response()
        except Exception as e:
            self._log_error(e, "_llm_respond_with_tools")
            # Return a fallback response
            if stream:
                return self._create_fallback_chat_stream()
            else:
                return self._create_fallback_chat_response()

    async def llm_respond(
        self,
        llm: lms.LLM,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = None,
        json_schema: Optional[JsonSchemaResponseFormat] = None,
        stream: Optional[bool] = False,
        tools: Optional[List[ToolDefinition]] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
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

            async def stream_generator() -> AsyncIterator[CompletionResponseStreamChunk]:
                try:
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
                except Exception as e:
                    self._log_error(e, "streaming completion")
                    # Return a fallback response in case of error
                    yield CompletionResponseStreamChunk(
                        delta="Error processing response",
                    )

            return stream_generator()
        else:
            try:
                response = await asyncio.to_thread(
                    llm.complete,
                    prompt=interleaved_content_as_str(content),
                    config=config,
                    response_format=json_schema,
                )

                # If we have a JSON schema, ensure the response is valid JSON
                if json_schema is not None:
                    valid_json = self._handle_json_extraction(response.content, "completion response")
                    if valid_json:
                        return CompletionResponse(
                            content=valid_json,  # Already serialized in _handle_json_extraction
                            stop_reason=self._get_stop_reason(response.stats.stop_reason),
                        )
                    # If we couldn't extract valid JSON, continue with the original content

                return CompletionResponse(
                    content=response.content,
                    stop_reason=self._get_stop_reason(response.stats.stop_reason),
                )
            except Exception as e:
                self._log_error(e, "LMStudio completion")
                # Return a fallback response with an error message
                return self._create_fallback_completion_response()

    def _convert_message_list_to_lmstudio_chat(self, messages: List[Message]) -> lms.Chat:
        chat = lms.Chat()
        for message in messages:
            if content_has_media(message.content):
                raise NotImplementedError("Media content is not supported in LMStudio messages")
            if message.role == "user":
                chat.add_user_message(interleaved_content_as_str(message.content))
            elif message.role == "system":
                chat.add_system_prompt(interleaved_content_as_str(message.content))
            elif message.role == "assistant":
                chat.add_assistant_response(interleaved_content_as_str(message.content))
            else:
                raise ValueError(f"Unsupported message role: {message.role}")
        return chat

    def _convert_prediction_to_chat_response(self, result: lms.PredictionResult) -> ChatCompletionResponse:
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
                    "topPSampling": params.strategy.top_p,
                }
            )
        elif isinstance(params.strategy, TopKSamplingStrategy):
            options.update({"topKSampling": params.strategy.top_k})
        else:
            raise ValueError(f"Unsupported sampling strategy: {params.strategy}")
        options.update(
            {
                "maxTokens": params.max_tokens if params.max_tokens != 0 else None,
                "repetitionPenalty": (params.repetition_penalty if params.repetition_penalty != 0 else None),
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
        """Asynchronously iterate over a synchronous iterable."""
        iterator = iter(iterable)

        def safe_next(it):
            """This is necessary to communicate StopIteration across threads"""
            try:
                return (next(it), False)
            except StopIteration:
                return (None, True)

        while True:
            item, done = await asyncio.to_thread(safe_next, iterator)
            if done:
                break
            yield item

    async def _convert_request_to_rest_call(self, request: ChatCompletionRequest) -> dict:
        compatible_request = self._convert_sampling_params(request.sampling_params)
        compatible_request["model"] = request.model
        compatible_request["messages"] = [await convert_message_to_openai_dict_new(m) for m in request.messages]
        if request.response_format:
            if isinstance(request.response_format, JsonSchemaResponseFormat):
                compatible_request["response_format"] = {
                    "type": "json_schema",
                    "json_schema": request.response_format.json_schema,
                }
            elif isinstance(request.response_format, GrammarResponseFormat):
                compatible_request["response_format"] = {
                    "type": "grammar",
                    "bnf": request.response_format.bnf,
                }
        if request.tools is not None:
            compatible_request["tools"] = [convert_tooldef_to_openai_tool(tool) for tool in request.tools]
        compatible_request["logprobs"] = False
        compatible_request["stream"] = request.stream
        compatible_request["extra_headers"] = {b"User-Agent": b"llama-stack: lmstudio-inference-adapter"}
        return compatible_request

    def _convert_sampling_params(self, sampling_params: Optional[SamplingParams]) -> dict:
        params: dict[str, Any] = {}

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
