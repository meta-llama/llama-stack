# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from llama_stack.apis.inference import (
    OpenAIChatCompletion,
    OpenAIChatCompletionChunk,
    OpenAIChoiceDelta,
    OpenAIChunkChoice,
    OpenAIMessageParam,
    OpenAIResponseFormatParam,
    OpenAISystemMessageParam,
)
from llama_stack.providers.remote.inference.groq.config import GroqConfig
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin
from llama_stack.providers.utils.inference.openai_compat import (
    prepare_openai_completion_params,
)

from .models import MODEL_ENTRIES


class GroqInferenceAdapter(LiteLLMOpenAIMixin):
    _config: GroqConfig

    def __init__(self, config: GroqConfig):
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=MODEL_ENTRIES,
            litellm_provider_name="groq",
            api_key_from_config=config.api_key,
            provider_data_api_key_field="groq_api_key",
        )
        self.config = config

    async def initialize(self):
        await super().initialize()

    async def shutdown(self):
        await super().shutdown()

    def _get_openai_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=f"{self.config.url}/openai/v1",
            api_key=self.get_api_key(),
        )

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
        model_obj = await self.model_store.get_model(model)

        # Groq does not support json_schema response format, so we need to convert it to json_object
        if response_format and response_format.type == "json_schema":
            response_format.type = "json_object"
            schema = response_format.json_schema.get("schema", {})
            response_format.json_schema = None
            json_instructions = f"\nYour response should be a JSON object that matches the following schema: {schema}"
            if messages and messages[0].role == "system":
                messages[0].content = messages[0].content + json_instructions
            else:
                messages.insert(0, OpenAISystemMessageParam(content=json_instructions))

        # Groq returns a 400 error if tools are provided but none are called
        # So, set tool_choice to "required" to attempt to force a call
        if tools and (not tool_choice or tool_choice == "auto"):
            tool_choice = "required"

        params = await prepare_openai_completion_params(
            model=model_obj.provider_resource_id,
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

        # Groq does not support streaming requests that set response_format
        fake_stream = False
        if stream and response_format:
            params["stream"] = False
            fake_stream = True

        response = await self._get_openai_client().chat.completions.create(**params)

        if fake_stream:
            chunk_choices = []
            for choice in response.choices:
                delta = OpenAIChoiceDelta(
                    content=choice.message.content,
                    role=choice.message.role,
                    tool_calls=choice.message.tool_calls,
                )
                chunk_choice = OpenAIChunkChoice(
                    delta=delta,
                    finish_reason=choice.finish_reason,
                    index=choice.index,
                    logprobs=None,
                )
                chunk_choices.append(chunk_choice)
            chunk = OpenAIChatCompletionChunk(
                id=response.id,
                choices=chunk_choices,
                object="chat.completion.chunk",
                created=response.created,
                model=response.model,
            )

            async def _fake_stream_generator():
                yield chunk

            return _fake_stream_generator()
        else:
            return response
