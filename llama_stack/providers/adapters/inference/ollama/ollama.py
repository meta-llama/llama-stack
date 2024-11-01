# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncGenerator

import httpx

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from ollama import AsyncClient

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.datatypes import ModelsProtocolPrivate

from llama_stack.providers.utils.inference.openai_compat import (
    get_sampling_options,
    OpenAICompatCompletionChoice,
    OpenAICompatCompletionResponse,
    process_chat_completion_response,
    process_chat_completion_stream_response,
    process_completion_response,
    process_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_prompt,
    completion_request_to_prompt,
)

OLLAMA_SUPPORTED_MODELS = {
    "Llama3.1-8B-Instruct": "llama3.1:8b-instruct-fp16",
    "Llama3.1-70B-Instruct": "llama3.1:70b-instruct-fp16",
    "Llama3.2-1B-Instruct": "llama3.2:1b-instruct-fp16",
    "Llama3.2-3B-Instruct": "llama3.2:3b-instruct-fp16",
    "Llama-Guard-3-8B": "llama-guard3:8b",
    "Llama-Guard-3-1B": "llama-guard3:1b",
}


class OllamaInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, url: str) -> None:
        self.url = url
        self.formatter = ChatFormat(Tokenizer.get_instance())

    @property
    def client(self) -> AsyncClient:
        return AsyncClient(host=self.url)

    async def initialize(self) -> None:
        print("Initializing Ollama, checking connectivity to server...")
        try:
            await self.client.ps()
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Ollama Server is not running, start it using `ollama serve` in a separate terminal"
            ) from e

    async def shutdown(self) -> None:
        pass

    async def register_model(self, model: ModelDef) -> None:
        raise ValueError("Dynamic model registration is not supported")

    async def list_models(self) -> List[ModelDef]:
        ollama_to_llama = {v: k for k, v in OLLAMA_SUPPORTED_MODELS.items()}

        ret = []
        res = await self.client.ps()
        for r in res["models"]:
            if r["model"] not in ollama_to_llama:
                print(f"Ollama is running a model unknown to Llama Stack: {r['model']}")
                continue

            llama_model = ollama_to_llama[r["model"]]
            ret.append(
                ModelDef(
                    identifier=llama_model,
                    llama_model=llama_model,
                    metadata={
                        "ollama_model": r["model"],
                    },
                )
            )

        return ret

    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        request = CompletionRequest(
            model=model,
            content=content,
            sampling_params=sampling_params,
            stream=stream,
            logprobs=logprobs,
        )
        if stream:
            return self._stream_completion(request)
        else:
            return await self._nonstream_completion(request)

    def _get_params_for_completion(self, request: CompletionRequest) -> dict:
        sampling_options = get_sampling_options(request.sampling_params)
        # This is needed since the Ollama API expects num_predict to be set
        # for early truncation instead of max_tokens.
        if sampling_options["max_tokens"] is not None:
            sampling_options["num_predict"] = sampling_options["max_tokens"]
        return {
            "model": OLLAMA_SUPPORTED_MODELS[request.model],
            "prompt": completion_request_to_prompt(request, self.formatter),
            "options": sampling_options,
            "raw": True,
            "stream": request.stream,
        }

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = self._get_params_for_completion(request)

        async def _generate_and_convert_to_openai_compat():
            s = await self.client.generate(**params)
            async for chunk in s:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=chunk["done_reason"] if chunk["done"] else None,
                    text=chunk["response"],
                )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_completion_stream_response(stream, self.formatter):
            yield chunk

    async def _nonstream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = self._get_params_for_completion(request)
        r = await self.client.generate(**params)
        assert isinstance(r, dict)

        choice = OpenAICompatCompletionChoice(
            finish_reason=r["done_reason"] if r["done"] else None,
            text=r["response"],
        )
        response = OpenAICompatCompletionResponse(
            choices=[choice],
        )

        return process_completion_response(response, self.formatter)

    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
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
            return self._stream_chat_completion(request)
        else:
            return await self._nonstream_chat_completion(request)

    def _get_params(self, request: ChatCompletionRequest) -> dict:
        return {
            "model": OLLAMA_SUPPORTED_MODELS[request.model],
            "prompt": chat_completion_request_to_prompt(request, self.formatter),
            "options": get_sampling_options(request.sampling_params),
            "raw": True,
            "stream": request.stream,
        }

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        params = self._get_params(request)
        r = await self.client.generate(**params)
        assert isinstance(r, dict)

        choice = OpenAICompatCompletionChoice(
            finish_reason=r["done_reason"] if r["done"] else None,
            text=r["response"],
        )
        response = OpenAICompatCompletionResponse(
            choices=[choice],
        )
        return process_chat_completion_response(response, self.formatter)

    async def _stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> AsyncGenerator:
        params = self._get_params(request)

        async def _generate_and_convert_to_openai_compat():
            s = await self.client.generate(**params)
            async for chunk in s:
                choice = OpenAICompatCompletionChoice(
                    finish_reason=chunk["done_reason"] if chunk["done"] else None,
                    text=chunk["response"],
                )
                yield OpenAICompatCompletionResponse(
                    choices=[choice],
                )

        stream = _generate_and_convert_to_openai_compat()
        async for chunk in process_chat_completion_stream_response(
            stream, self.formatter
        ):
            yield chunk

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
