# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from llama_stack.apis.inference.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAICompletion,
    OpenAIEmbeddingData,
    OpenAIEmbeddingsResponse,
    OpenAIEmbeddingUsage,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
)
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_compat import prepare_openai_completion_params

from .config import OpenAIConfig
from .models import MODEL_ENTRIES

logger = logging.getLogger(__name__)


#
# This OpenAI adapter implements Inference methods using two clients -
#
# | Inference Method           | Implementation Source    |
# |----------------------------|--------------------------|
# | completion                 | LiteLLMOpenAIMixin       |
# | chat_completion            | LiteLLMOpenAIMixin       |
# | embedding                  | LiteLLMOpenAIMixin       |
# | batch_completion           | LiteLLMOpenAIMixin       |
# | batch_chat_completion      | LiteLLMOpenAIMixin       |
# | openai_completion          | AsyncOpenAI              |
# | openai_chat_completion     | AsyncOpenAI              |
# | openai_embeddings          | AsyncOpenAI              |
#
class OpenAIInferenceAdapter(LiteLLMOpenAIMixin):
    def __init__(self, config: OpenAIConfig) -> None:
        LiteLLMOpenAIMixin.__init__(
            self,
            MODEL_ENTRIES,
            api_key_from_config=config.api_key,
            provider_data_api_key_field="openai_api_key",
        )
        self.config = config
        # we set is_openai_compat so users can use the canonical
        # openai model names like "gpt-4" or "gpt-3.5-turbo"
        # and the model name will be translated to litellm's
        # "openai/gpt-4" or "openai/gpt-3.5-turbo" transparently.
        # if we do not set this, users will be exposed to the
        # litellm specific model names, an abstraction leak.
        self.is_openai_compat = True
        self._openai_client = AsyncOpenAI(
            api_key=self.config.api_key,
        )

    async def initialize(self) -> None:
        await super().initialize()

    async def shutdown(self) -> None:
        await super().shutdown()

    async def openai_completion(
        self,
        model: str,
        prompt: str | list[str] | list[int] | list[list[int]],
        best_of: int | None = None,
        echo: bool | None = None,
        frequency_penalty: float | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        user: str | None = None,
        guided_choice: list[str] | None = None,
        prompt_logprobs: int | None = None,
        suffix: str | None = None,
    ) -> OpenAICompletion:
        if guided_choice is not None:
            logging.warning("guided_choice is not supported by the OpenAI API. Ignoring.")
        if prompt_logprobs is not None:
            logging.warning("prompt_logprobs is not supported by the OpenAI API. Ignoring.")

        model_id = (await self.model_store.get_model(model)).provider_resource_id
        if model_id.startswith("openai/"):
            model_id = model_id[len("openai/") :]
        params = await prepare_openai_completion_params(
            model=model_id,
            prompt=prompt,
            best_of=best_of,
            echo=echo,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_tokens=max_tokens,
            n=n,
            presence_penalty=presence_penalty,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            top_p=top_p,
            user=user,
            suffix=suffix,
        )
        return await self._openai_client.completions.create(**params)

    async def openai_chat_completion(
        self,
        model: str,
        messages: list[OpenAIMessageParam],
        frequency_penalty: float | None = None,
        function_call: str | dict[str, Any] | None = None,
        functions: list[dict[str, Any]] | None = None,
        logit_bias: dict[str, float] | None = None,
        logprobs: bool | None = None,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        presence_penalty: float | None = None,
        response_format: OpenAIResponseFormatParam | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stream: bool | None = None,
        stream_options: dict[str, Any] | None = None,
        temperature: float | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        top_logprobs: int | None = None,
        top_p: float | None = None,
        user: str | None = None,
    ) -> OpenAIChatCompletion | AsyncIterator[OpenAIChatCompletionChunk]:
        model_id = (await self.model_store.get_model(model)).provider_resource_id
        if model_id.startswith("openai/"):
            model_id = model_id[len("openai/") :]
        params = await prepare_openai_completion_params(
            model=model_id,
            messages=messages,
            frequency_penalty=frequency_penalty,
            function_call=function_call,
            functions=functions,
            logit_bias=logit_bias,
            logprobs=logprobs,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            presence_penalty=presence_penalty,
            response_format=response_format,
            seed=seed,
            stop=stop,
            stream=stream,
            stream_options=stream_options,
            temperature=temperature,
            tool_choice=tool_choice,
            tools=tools,
            top_logprobs=top_logprobs,
            top_p=top_p,
            user=user,
        )
        return await self._openai_client.chat.completions.create(**params)

    async def openai_embeddings(
        self,
        model: str,
        input: str | list[str],
        encoding_format: str | None = "float",
        dimensions: int | None = None,
        user: str | None = None,
    ) -> OpenAIEmbeddingsResponse:
        model_id = (await self.model_store.get_model(model)).provider_resource_id
        if model_id.startswith("openai/"):
            model_id = model_id[len("openai/") :]

        # Prepare parameters for OpenAI embeddings API
        params = {
            "model": model_id,
            "input": input,
        }

        if encoding_format is not None:
            params["encoding_format"] = encoding_format
        if dimensions is not None:
            params["dimensions"] = dimensions
        if user is not None:
            params["user"] = user

        # Call OpenAI embeddings API
        response = await self._openai_client.embeddings.create(**params)

        data = []
        for i, embedding_data in enumerate(response.data):
            data.append(
                OpenAIEmbeddingData(
                    embedding=embedding_data.embedding,
                    index=i,
                )
            )

        usage = OpenAIEmbeddingUsage(
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
        )

        return OpenAIEmbeddingsResponse(
            data=data,
            model=response.model,
            usage=usage,
        )
