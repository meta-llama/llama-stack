# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Union

from llama_models.datatypes import CoreModelId
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer
from openai import AsyncOpenAI

from llama_stack.apis.inference import (
    AsyncIterator,
    ChatCompletionRequest,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseStreamChunk,
    EmbeddingsResponse,
    Inference,
    InterleavedTextMedia,
    LogProbConfig,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
)
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_model_alias,
)
from llama_stack.providers.utils.inference.openai_compat import (
    ChatCompletionResponseStreamChunk,
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
)

from .config import SambanovaImplConfig


class SambanovaErrorCode(str, Enum):
    INVALID_AUTH = "invalid_authentication"
    REQUEST_TIMEOUT = "request_timeout"
    INSUFFICIENT_QUOTA = "insufficient_quota"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    INVALID_TYPE = "invalid_type"
    MODEL_NOT_FOUND = "model_not_found"
    VALUE_ABOVE_MAX = "decimal_above_max_value"
    VALUE_BELOW_MIN = "decimal_below_min_value"
    INTEGER_ABOVE_MAX = "integer_above_max_value"


MODEL_ALIASES = [
    build_model_alias(
        "Meta-Llama-3.2-1B-Instruct",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_alias(
        "Meta-Llama-3.2-3B-Instruct",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "Llama-3.2-11B-Vision-Instruct",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "Llama-3.2-90B-Vision-Instruct",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_model_alias(
        "Meta-Llama-3.1-8B-Instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "Meta-Llama-3.1-70B-Instruct",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "Meta-Llama-3.1-405B-Instruct",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
]

FUNCTION_CALLING_MODELS = {
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3.1-70B-Instruct",
    "Meta-Llama-3.1-405B-Instruct",
}

UNSUPPORTED_PARAMS = {
    "logprobs",
    "top_logprobs",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "parallel_tool_calls",
    "seed",
    "response_format",
}


class SambanovaInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):
    """SambaNova inference adapter using OpenAI client compatibility layer.

    This adapter provides access to SambaNova's AI models through their OpenAI-compatible API.
    It handles authentication, request formatting, and response processing while managing
    unsupported features gracefully.

    Note: Some OpenAI features are not supported:
        - logprobs, top_logprobs, n
        - presence_penalty, frequency_penalty
        - logit_bias
        - tools and tool_choice (function calling)
        - parallel_tool_calls, seed
        - stream_options
        - response_format (JSON mode)
    """

    def __init__(self, config: SambanovaImplConfig) -> None:
        """Initialize the SambaNova inference adapter.

        Args:
            config: Configuration for the SambaNova implementation
        """
        ModelRegistryHelper.__init__(self, MODEL_ALIASES)
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())
        self._client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> AsyncOpenAI:
        """Get or create the OpenAI client instance.

        Returns:
            AsyncOpenAI: The configured client instance
        """
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url="https://api.sambanova.ai/v1",
                api_key=self._get_api_key(),
                timeout=60.0,
            )
        return self._client

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    def _get_api_key(self) -> str:
        """Get the API key from config or request headers.

        Returns:
            str: The API key to use

        Raises:
            ValueError: If no API key is available
        """
        if self.config.api_key is not None:
            return self.config.api_key

        provider_data = self.get_request_provider_data()
        if provider_data is None or not provider_data.sambanova_api_key:
            raise ValueError(
                'Pass SambaNova API Key in the header X-LlamaStack-ProviderData as { "sambanova_api_key": <your api key>}'
            )
        return provider_data.sambanova_api_key

    def _filter_unsupported_params(self, params: Dict) -> Dict:
        """Remove parameters not supported by SambaNova API.

        Args:
            params: Original parameters dictionary

        Returns:
            Dict: Filtered parameters dictionary
        """
        return {k: v for k, v in params.items() if k not in UNSUPPORTED_PARAMS}

    async def _get_params(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        """Prepare parameters for the API request.

        Args:
            request: The completion request

        Returns:
            dict: Prepared parameters for the API call
        """
        # Get and process sampling options
        sampling_options = get_sampling_options(request.sampling_params)
        filtered_options = self._filter_unsupported_params(sampling_options)

        if "temperature" in filtered_options:
            filtered_options["temperature"] = min(
                max(filtered_options["temperature"], 0), 1
            )

        input_dict = {}
        if isinstance(request, ChatCompletionRequest):
            input_dict["messages"] = [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ]

            if request.tools and self._supports_function_calling(request.model):
                input_dict["tools"] = [
                    self._convert_tool_to_function(tool)
                    for tool in request.tools
                ]

                if request.tool_choice == ToolChoice.auto:
                    input_dict["tool_choice"] = "auto"
                elif request.tool_choice == ToolChoice.required:
                    input_dict["tool_choice"] = "required"
                elif isinstance(request.tool_choice, str):
                    input_dict["tool_choice"] = {
                        "type": "function",
                        "function": {"name": request.tool_choice},
                    }
        else:
            input_dict["prompt"] = request.content

        return {
            "model": request.model,
            **input_dict,
            **filtered_options,
            "stream": request.stream,
        }

    async def _handle_sambanova_error(self, e: Exception) -> None:
        """Handle SambaNova specific API errors with detailed messages.

        Args:
            e: The exception to handle

        Raises:
            ValueError: For client errors
            RuntimeError: For server errors
        """
        error_msg = str(e)
        error_data = {}

        try:
            if hasattr(e, "response"):
                error_data = e.response.json().get("error", {})
        except Exception:
            pass

        error_code = error_data.get("code", "")
        error_message = error_data.get("message", error_msg)
        error_param = error_data.get("param", "")

        if "401" in error_msg or error_code == SambanovaErrorCode.INVALID_AUTH:
            raise ValueError("Invalid API key or unauthorized access") from e

        elif (
            "408" in error_msg
            or error_code == SambanovaErrorCode.REQUEST_TIMEOUT
        ):
            raise ValueError(
                "Request timed out. Consider upgrading to a higher tier offering"
            ) from e

        elif (
            "429" in error_msg
            or error_code == SambanovaErrorCode.INSUFFICIENT_QUOTA
        ):
            raise ValueError(
                "Rate limit exceeded. Consider upgrading to a higher tier offering"
            ) from e

        elif "400" in error_msg:
            if error_code == SambanovaErrorCode.CONTEXT_LENGTH_EXCEEDED:
                raise ValueError(
                    "Total number of input and output tokens exceeds model's context length"
                ) from e

            elif error_code == SambanovaErrorCode.INVALID_TYPE:
                raise ValueError(
                    f"Invalid parameter type for {error_param}: {error_message}"
                ) from e

            elif error_code in (
                SambanovaErrorCode.VALUE_ABOVE_MAX,
                SambanovaErrorCode.VALUE_BELOW_MIN,
                SambanovaErrorCode.INTEGER_ABOVE_MAX,
            ):
                raise ValueError(
                    f"Invalid value for {error_param}: {error_message}"
                ) from e

            elif error_code == SambanovaErrorCode.MODEL_NOT_FOUND:
                raise ValueError(f"Model not found: {error_message}") from e

            else:
                raise ValueError(f"Bad request: {error_message}") from e

        raise RuntimeError(f"SambaNova API error: {error_message}") from e

    def _supports_function_calling(self, model: str) -> bool:
        """Check if the model supports function calling.

        Args:
            model: Model name to check

        Returns:
            bool: True if model supports function calling
        """
        return any(
            model.startswith(supported) for supported in FUNCTION_CALLING_MODELS
        )

    def _convert_tool_to_function(self, tool: ToolDefinition) -> dict:
        """Convert a ToolDefinition to SambaNova function format.

        Args:
            tool: Tool definition to convert

        Returns:
            dict: Function definition in SambaNova format
        """
        return {
            "type": "function",
            "function": {
                "name": tool.tool_name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        name: {
                            "type": param.param_type,
                            "description": param.description,
                        }
                        for name, param in tool.parameters.items()
                    },
                    "required": list(tool.parameters.keys()),
                },
            },
        }

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        try:
            params = await self._get_params(request)
            response = await self.client.chat.completions.create(**params)

            if (
                self._supports_function_calling(request.model)
                and response.choices[0].message.tool_calls
            ):
                tool_call = response.choices[0].message.tool_calls[0]
                choice = OpenAICompatCompletionChoice(
                    finish_reason=response.choices[0].finish_reason,
                    text="",
                    tool_calls=[
                        {
                            "tool_name": tool_call.function.name,
                            "arguments": tool_call.function.arguments or "",
                        }
                    ],
                )
            else:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=response.choices[0].finish_reason,
                    text=response.choices[0].message.content or "",
                    tool_calls=[],
                )

            compat_response = OpenAICompatCompletionResponse(choices=[choice])
            return process_chat_completion_response(
                compat_response, self.formatter
            )

        except Exception as e:
            await self._handle_sambanova_error(e)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionResponseStreamChunk]:
        try:
            params = await self._get_params(request)
            stream = await self.client.chat.completions.create(**params)

            async def _to_async_generator():
                async for chunk in stream:
                    if (
                        self._supports_function_calling(request.model)
                        and chunk.choices[0].delta.tool_calls
                    ):
                        tool_call = chunk.choices[0].delta.tool_calls[0]
                        choice = OpenAICompatCompletionChoice(
                            finish_reason=chunk.choices[0].finish_reason,
                            text="",
                            tool_calls=[
                                {
                                    "tool_name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments
                                    or "",
                                }
                            ]
                            if tool_call.function
                            else None,
                        )
                    else:
                        choice = OpenAICompatCompletionChoice(
                            finish_reason=chunk.choices[0].finish_reason,
                            text=chunk.choices[0].delta.content or "",
                            tool_calls=[],
                        )
                    yield OpenAICompatCompletionResponse(choices=[choice])

            async for chunk in process_chat_completion_stream_response(
                _to_async_generator(), self.formatter
            ):
                yield chunk

        except Exception as e:
            await self._handle_sambanova_error(e)

    def completion(
        self,
        model_id: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[
        CompletionResponse, AsyncIterator[CompletionResponseStreamChunk]
    ]:
        raise NotImplementedError("SambaNova does not support text completion")

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
        """Handle chat completion requests.

        Args:
            model_id: The model identifier
            messages: List of chat messages
            sampling_params: Parameters for text generation
            tools: Tool definitions (supported only for specific models)
            tool_choice: Tool choice option
            tool_prompt_format: Tool prompt format
            response_format: Response format (not supported)
            stream: Whether to stream the response
            logprobs: Log probability config (not supported)

        Returns:
            AsyncGenerator: The completion response

        Raises:
            ValueError: If function calling is requested for unsupported model
        """
        model = await self.model_store.get_model(model_id)

        # Raise error for tool usage with unsupported models
        if tools and not self._supports_function_calling(
            model.provider_resource_id
        ):
            raise ValueError(
                f"Function calling is not supported for model {model.provider_resource_id}. "
                f"Only the following models support function calling: "
                f"{', '.join(FUNCTION_CALLING_MODELS)}"
            )

        request = ChatCompletionRequest(
            model=model.provider_resource_id,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        """Embeddings are not supported.

        Raises:
            NotImplementedError: Always raised as this feature is not supported
        """
        raise NotImplementedError("Embeddings not supported for SambaNova")
