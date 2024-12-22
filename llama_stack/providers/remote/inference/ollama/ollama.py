# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import AsyncGenerator

import httpx
from llama_models.datatypes import CoreModelId

from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer
from ollama import AsyncClient

from llama_stack.providers.utils.inference.model_registry import (
    build_model_alias,
    build_model_alias_with_just_provider_model_id,
    ModelRegistryHelper,
)

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.common.content_types import ImageContentItem, TextContentItem
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
    content_has_media,
    convert_image_content_to_url,
    interleaved_content_as_str,
    request_has_media,
)

log = logging.getLogger(__name__)

model_aliases = [
    build_model_alias(
        "llama3.1:8b-instruct-fp16",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias_with_just_provider_model_id(
        "llama3.1:8b",
        CoreModelId.llama3_1_8b_instruct.value,
    ),
    build_model_alias(
        "llama3.1:70b-instruct-fp16",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias_with_just_provider_model_id(
        "llama3.1:70b",
        CoreModelId.llama3_1_70b_instruct.value,
    ),
    build_model_alias(
        "llama3.1:405b-instruct-fp16",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_alias_with_just_provider_model_id(
        "llama3.1:405b",
        CoreModelId.llama3_1_405b_instruct.value,
    ),
    build_model_alias(
        "llama3.2:1b-instruct-fp16",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_alias_with_just_provider_model_id(
        "llama3.2:1b",
        CoreModelId.llama3_2_1b_instruct.value,
    ),
    build_model_alias(
        "llama3.2:3b-instruct-fp16",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias_with_just_provider_model_id(
        "llama3.2:3b",
        CoreModelId.llama3_2_3b_instruct.value,
    ),
    build_model_alias(
        "llama3.2-vision:11b-instruct-fp16",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias_with_just_provider_model_id(
        "llama3.2-vision:latest",
        CoreModelId.llama3_2_11b_vision_instruct.value,
    ),
    build_model_alias(
        "llama3.2-vision:90b-instruct-fp16",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_model_alias_with_just_provider_model_id(
        "llama3.2-vision:90b",
        CoreModelId.llama3_2_90b_vision_instruct.value,
    ),
    build_model_alias(
        "llama3.3:70b",
        CoreModelId.llama3_3_70b_instruct.value,
    ),
    # The Llama Guard models don't have their full fp16 versions
    # so we are going to alias their default version to the canonical SKU
    build_model_alias(
        "llama-guard3:8b",
        CoreModelId.llama_guard_3_8b.value,
    ),
    build_model_alias(
        "llama-guard3:1b",
        CoreModelId.llama_guard_3_1b.value,
    ),
]


class OllamaInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, url: str) -> None:
        self.register_helper = ModelRegistryHelper(model_aliases)
        self.url = url
        self.formatter = ChatFormat(Tokenizer.get_instance())

    @property
    def client(self) -> AsyncClient:
        return AsyncClient(host=self.url)

    async def initialize(self) -> None:
        log.info(f"checking connectivity to Ollama at `{self.url}`...")
        try:
            await self.client.ps()
        except httpx.ConnectError as e:
            raise RuntimeError(
                "Ollama Server is not running, start it using `ollama serve` in a separate terminal"
            ) from e

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

    async def _stream_completion(self, request: CompletionRequest) -> AsyncGenerator:
        params = await self._get_params(request)

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
        params = await self._get_params(request)
        r = await self.client.generate(**params)

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
        model_id: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = ToolChoice.auto,
        tool_prompt_format: Optional[ToolPromptFormat] = ToolPromptFormat.json,
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

    async def _get_params(
        self, request: Union[ChatCompletionRequest, CompletionRequest]
    ) -> dict:
        sampling_options = get_sampling_options(request.sampling_params)
        # This is needed since the Ollama API expects num_predict to be set
        # for early truncation instead of max_tokens.
        if sampling_options.get("max_tokens") is not None:
            sampling_options["num_predict"] = sampling_options["max_tokens"]

        input_dict = {}
        media_present = request_has_media(request)
        if isinstance(request, ChatCompletionRequest):
            if media_present:
                contents = [
                    await convert_message_to_openai_dict_for_ollama(m)
                    for m in request.messages
                ]
                # flatten the list of lists
                input_dict["messages"] = [
                    item for sublist in contents for item in sublist
                ]
            else:
                input_dict["raw"] = True
                input_dict["prompt"] = await chat_completion_request_to_prompt(
                    request,
                    self.register_helper.get_llama_model(request.model),
                    self.formatter,
                )
        else:
            assert (
                not media_present
            ), "Ollama does not support media for Completion requests"
            input_dict["prompt"] = await completion_request_to_prompt(
                request, self.formatter
            )
            input_dict["raw"] = True

        return {
            "model": request.model,
            **input_dict,
            "options": sampling_options,
            "stream": request.stream,
        }

    async def _nonstream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        params = await self._get_params(request)
        if "messages" in params:
            r = await self.client.chat(**params)
        else:
            r = await self.client.generate(**params)

        if "message" in r:
            choice = OpenAICompatCompletionChoice(
                finish_reason=r["done_reason"] if r["done"] else None,
                text=r["message"]["content"],
            )
        else:
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
        params = await self._get_params(request)

        async def _generate_and_convert_to_openai_compat():
            if "messages" in params:
                s = await self.client.chat(**params)
            else:
                s = await self.client.generate(**params)
            async for chunk in s:
                if "message" in chunk:
                    choice = OpenAICompatCompletionChoice(
                        finish_reason=chunk["done_reason"] if chunk["done"] else None,
                        text=chunk["message"]["content"],
                    )
                else:
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
        model_id: str,
        contents: List[InterleavedContent],
    ) -> EmbeddingsResponse:
        model = await self.model_store.get_model(model_id)

        assert all(
            not content_has_media(content) for content in contents
        ), "Ollama does not support media for embeddings"
        response = await self.client.embed(
            model=model.provider_resource_id,
            input=[interleaved_content_as_str(content) for content in contents],
        )
        embeddings = response["embeddings"]

        return EmbeddingsResponse(embeddings=embeddings)

    async def register_model(self, model: Model) -> Model:
        # ollama does not have embedding models running. Check if the model is in list of available models.
        if model.model_type == ModelType.embedding:
            response = await self.client.list()
            available_models = [m["model"] for m in response["models"]]
            if model.provider_resource_id not in available_models:
                raise ValueError(
                    f"Model '{model.provider_resource_id}' is not available in Ollama. "
                    f"Available models: {', '.join(available_models)}"
                )
            return model
        model = await self.register_helper.register_model(model)
        models = await self.client.ps()
        available_models = [m["model"] for m in models["models"]]
        if model.provider_resource_id not in available_models:
            raise ValueError(
                f"Model '{model.provider_resource_id}' is not available in Ollama. "
                f"Available models: {', '.join(available_models)}"
            )

        return model


async def convert_message_to_openai_dict_for_ollama(message: Message) -> List[dict]:
    async def _convert_content(content) -> dict:
        if isinstance(content, ImageContentItem):
            return {
                "role": message.role,
                "images": [
                    await convert_image_content_to_url(
                        content, download=True, include_format=False
                    )
                ],
            }
        else:
            text = content.text if isinstance(content, TextContentItem) else content
            assert isinstance(text, str)
            return {
                "role": message.role,
                "content": text,
            }

    if isinstance(message.content, list):
        return [await _convert_content(c) for c in message.content]
    else:
        return [await _convert_content(message.content)]
