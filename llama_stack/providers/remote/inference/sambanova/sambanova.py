from typing import AsyncGenerator, List, Optional, Union

import httpx
from llama_models.datatypes import CoreModelId
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from llama_stack.apis.inference import *
from llama_stack.distribution.request_headers import NeedsRequestProviderData
from llama_stack.providers.utils.inference.model_registry import (
    ModelRegistryHelper,
    build_model_alias,
)
from llama_stack.providers.utils.inference.openai_compat import (
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    get_sampling_options,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    completion_request_to_prompt,
)

from .config import SambanovaImplConfig

# Simplified model aliases - focus on core models
MODEL_ALIASES = [
    build_model_alias(
        "Meta-Llama-3.1-8B-Instruct",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
]


class SambanovaInferenceAdapter(
    ModelRegistryHelper, Inference, NeedsRequestProviderData
):
    def __init__(self, config: SambanovaImplConfig) -> None:
        ModelRegistryHelper.__init__(self, MODEL_ALIASES)
        self.config = config
        self.formatter = ChatFormat(Tokenizer.get_instance())
        self.client = httpx.AsyncClient(
            base_url=self.config.url,
            timeout=httpx.Timeout(timeout=300.0),
        )

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        await self.client.aclose()

    def _get_api_key(self) -> str:
        if self.config.api_key is not None:
            return self.config.api_key

        provider_data = self.get_request_provider_data()
        if provider_data is None or not provider_data.sambanova_api_key:
            raise ValueError(
                'Pass SambaNova API Key in the header X-LlamaStack-ProviderData as { "sambanova_api_key": <your api key>}'
            )
        return provider_data.sambanova_api_key

    def _convert_messages_to_api_format(self, messages: List[Message]) -> List[dict]:
        """Convert our Message objects to SambaNova API format."""
        return [
            {"role": message.role, "content": message.content} for message in messages
        ]

    def _get_sampling_params(self, params: Optional[SamplingParams]) -> dict:
        """Convert our SamplingParams to SambaNova API parameters."""
        if not params:
            return {}

        api_params = {}
        if params.max_tokens:
            api_params["max_tokens"] = params.max_tokens
        if params.temperature is not None:
            api_params["temperature"] = params.temperature
        if params.top_p is not None:
            api_params["top_p"] = params.top_p
        if params.top_k is not None:
            api_params["top_k"] = params.top_k
        if params.stop_sequences:
            api_params["stop"] = params.stop_sequences

        return api_params

    async def completion(
        self,
        model_id: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        model = await self.model_store.get_model(model_id)
        request = CompletionRequest(
            model=model.provider_resource_id,
            content=content,
            sampling_params=sampling_params,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    async def _get_params(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        sampling_options = get_sampling_options(request.sampling_params)

        input_dict = {}
        if isinstance(request, ChatCompletionRequest):
            if isinstance(request.messages[0].content, list):
                raise NotImplementedError("Media content not supported for SambaNova")
            input_dict["messages"] = self._convert_messages_to_api_format(
                request.messages
            )
        else:
            input_dict["prompt"] = completion_request_to_prompt(request, self.formatter)

        return {
            "model": request.model,
            **input_dict,
            **sampling_options,
            "stream": request.stream,
        }

    async def _nonstream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)
        try:
            response = await self.client.post(
                "/completions",
                json=params,
                headers={"Authorization": f"Bearer {self._get_api_key()}"},
            )
            response.raise_for_status()
            data = response.json()

            choice = OpenAICompatCompletionChoice(
                finish_reason=data.get("choices", [{}])[0].get("finish_reason"),
                text=data.get("choices", [{}])[0].get("text", ""),
            )
            response = OpenAICompatCompletionResponse(
                choices=[choice],
            )
            return process_completion_response(response, self.formatter)
        except httpx.HTTPError as e:
            await self._handle_api_error(e)

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _to_async_generator():
            try:
                async with self.client.stream(
                    "POST",
                    "/completions",
                    json=params,
                    headers={"Authorization": f"Bearer {self._get_api_key()}"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = httpx.loads(line)
                            choice = OpenAICompatCompletionChoice(
                                finish_reason=data.get("choices", [{}])[0].get(
                                    "finish_reason"
                                ),
                                text=data.get("choices", [{}])[0].get("text", ""),
                            )
                            yield OpenAICompatCompletionResponse(choices=[choice])
            except httpx.HTTPError as e:
                await self._handle_api_error(e)

        stream = _to_async_generator()
        async for chunk in process_completion_stream_response(stream, self.formatter):
            yield chunk

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
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        params = await self._get_params(request)
        try:
            response = await self.client.post(
                "/chat/completions",
                json=params,
                headers={"Authorization": f"Bearer {self._get_api_key()}"},
            )
            response.raise_for_status()
            data = response.json()

            choice = OpenAICompatCompletionChoice(
                finish_reason=data.get("choices", [{}])[0].get("finish_reason"),
                text=data.get("choices", [{}])[0].get("message", {}).get("content", ""),
            )
            response = OpenAICompatCompletionResponse(choices=[choice])
            return process_chat_completion_response(response, self.formatter)
        except httpx.HTTPError as e:
            await self._handle_api_error(e)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        params = await self._get_params(request)

        async def _to_async_generator():
            try:
                async with self.client.stream(
                    "POST",
                    "/chat/completions",
                    json=params,
                    headers={"Authorization": f"Bearer {self._get_api_key()}"},
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = httpx.loads(line)
                            choice = OpenAICompatCompletionChoice(
                                finish_reason=data.get("choices", [{}])[0].get(
                                    "finish_reason"
                                ),
                                text=data.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", ""),
                            )
                            yield OpenAICompatCompletionResponse(choices=[choice])
            except httpx.HTTPError as e:
                await self._handle_api_error(e)

        stream = _to_async_generator()
        async for chunk in process_chat_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk

    async def _handle_api_error(self, e: httpx.HTTPError) -> None:
        if e.response.status_code in (401, 403):
            raise ValueError("Invalid API key or unauthorized access") from e
        elif e.response.status_code == 429:
            raise ValueError("Rate limit exceeded") from e
        elif e.response.status_code == 400:
            error_data = e.response.json()
            raise ValueError(
                f"Bad request: {error_data.get('error', {}).get('message', 'Unknown error')}"
            ) from e
        raise RuntimeError(f"SambaNova API error: {str(e)}") from e

    async def embeddings(
        self,
        model_id: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError("Embeddings not supported for SambaNova")
