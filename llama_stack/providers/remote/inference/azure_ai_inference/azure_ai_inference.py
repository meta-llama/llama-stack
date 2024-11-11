# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
from typing import AsyncGenerator

from llama_models.llama3.api.chat_format import ChatFormat

from llama_models.llama3.api.datatypes import Message
from llama_models.llama3.api.tokenizer import Tokenizer

from azure.ai.inference.aio import ChatCompletionsClient as ChatCompletionsClientAsync
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.identity import DefaultAzureCredential

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.providers.datatypes import ModelsProtocolPrivate

from llama_stack.providers.utils.inference.openai_compat import (
    process_chat_completion_response,
    process_chat_completion_stream_response,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_messages,
)

from .config import AzureAIInferenceConfig

# Mapping of model names from the Llama model names to the Azure AI model catalog names
SUPPORTED_INSTRUCT_MODELS = {
    "Llama3.1-8B-Instruct": "Meta-Llama-3.1-8B-Instruct",
    "Llama3.1-70B-Instruct": "Meta-Llama-3.1-70B-Instruct",
    "Llama3.1-405B-Instruct": "Meta-Llama-3.1-405B-Instruct",
    "Llama3.2-1B-Instruct": "Llama-3.2-1B-Instruct",
    "Llama3.2-3B-Instruct": "Llama-3.2-3B-Instruct",
    "Llama3.2-11B-Vision-Instruct": "Llama-3.2-11B-Vision-Instruct",
    "Llama3.2-90B-Vision-Instruct": "Llama-3.2-90B-Vision-Instruct",
}

logger = logging.getLogger(__name__)

class AzureAIInferenceAdapter(Inference, ModelsProtocolPrivate):
    def __init__(self, config: AzureAIInferenceConfig) -> None:
        tokenizer = Tokenizer.get_instance()

        self.config = config
        self.formatter = ChatFormat(tokenizer)
        self._model_name = None
        
    @property
    def client(self) -> ChatCompletionsClientAsync:
        if self.config.credential is None:
             credential = DefaultAzureCredential()
        else:
            credential = AzureKeyCredential(self.config.credential)

        if self.config.api_version:
            return ChatCompletionsClientAsync(
                endpoint=self.config.endpoint, 
                credential=credential,
                user_agent="llama-stack",
                api_version=self.config.api_version,
            )
        else:
            return ChatCompletionsClientAsync(
                endpoint=self.config.endpoint,
                credential=credential,
                user_agent="llama-stack",
            )

    async def initialize(self) -> None:
        async with self.client as async_client:
            try:
                model_info = await async_client.get_model_info()
                if model_info:
                    self._model_name = model_info.get("model_name", None)
                    logger.info(
                        f"Endpoint {self.config.endpoint} supports model {self._model_name}"
                    )
                if self._model_name not in SUPPORTED_INSTRUCT_MODELS.values():
                    logger.warning(
                        f"Endpoints serves model {self._model_name} which may not be supported"
                    )
            except HttpResponseError:
                logger.info(
                    f"Endpoint {self.config.endpoint} supports multiple models"
                )
                self._model_name = None


    async def shutdown(self) -> None:
        pass


    async def list_models(self) -> List[ModelDef]:
        if self._model_name is None:
            return [
                ModelDef(identifier=model_name, llama_model=azure_model_id)
                for model_name, azure_model_id in SUPPORTED_INSTRUCT_MODELS.items()
            ]
        else:
            # find if there is a value in the SUPPORTED_INSTRUCT_MODELS that matches the model name
            supported_model = next(
                (model for model in SUPPORTED_INSTRUCT_MODELS if SUPPORTED_INSTRUCT_MODELS[model] == self._model_name), 
                None
            )
            return [
                ModelDef(
                    identifier=supported_model or self._model_name,
                    llama_model=self._model_name
                )
            ]


    async def completion(
        self,
        model: str,
        content: InterleavedTextMedia,
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncGenerator:
        raise NotImplementedError()


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
            model=model or self.config.model_name,
            messages=messages,
            sampling_params=sampling_params,
            tools=tools or [],
            tool_choice=tool_choice,
            tool_prompt_format=tool_prompt_format,
            stream=stream,
            logprobs=logprobs,
        )
        params = self._get_params(request)
        if stream:
            return self._stream_chat_completion(params)
        else:
            return await self._nonstream_chat_completion(params)


    async def _nonstream_chat_completion(
        self, params: dict
    ) -> ChatCompletionResponse:
        async with self.client as client:
            r = await client.complete(**params)
            return process_chat_completion_response(r, self.formatter)


    async def _stream_chat_completion(
        self, params: dict
    ) -> AsyncGenerator:
        async with self.client as client:
            stream = await client.complete(**params, stream=True)
            async for chunk in process_chat_completion_stream_response(
                stream, self.formatter
            ):
                yield chunk


    @staticmethod
    def _get_sampling_options(
        params: SamplingParams, 
        logprobs: Optional[LogProbConfig] = None
    ) -> dict:
        options = {}
        model_extras = {}
        if params:
            # repetition_penalty is not supported by Azure AI inference API
            for attr in {"temperature", "top_p", "max_tokens"}:
                if getattr(params, attr):
                    options[attr] = getattr(params, attr)

            if params.top_k is not None and params.top_k != 0:
                model_extras["top_k"] = params.top_k

            if logprobs is not None:
                model_extras["logprobs"] = params.logprobs

            if model_extras:
                options["model_extras"] = model_extras

        return options

    @staticmethod
    def _to_azure_ai_messages(messages: List[Message]) -> List[dict]:
        """
        Convert the messages to the format expected by the Azure AI API.
        """
        azure_ai_messages = []
        for message in messages:
            role = message.role
            content = message.content

            if role == "user":
                azure_ai_messages.append({"role": role, "content": content})
            elif role == "assistant":
                azure_ai_messages.append({"role": role, "content": content, "tool_calls": message.tool_calls})
            elif role == "system":
                azure_ai_messages.append({"role": role, "content": content})
            elif role == "ipython":
                azure_ai_messages.append(
                    {
                        "role": "tool", 
                        "content": content,
                        "tool_call_id": message.call_id
                    }
                )

        return azure_ai_messages
        

    def _get_params(self, request: ChatCompletionRequest) -> dict:
        """
        Gets the parameters for the Azure AI model inference API from the Chat completions request.
        Parameters are returned as a dictionary.
        """
        options = self._get_sampling_options(request.sampling_params, request.logprobs)
        messages = self._to_azure_ai_messages(chat_completion_request_to_messages(request))
        if (self._model_name):
            # If the model name is already resolved, then the endpoint
            # is serving a single model and we don't need to specify it
            return {
                "messages": messages,
                **options
            }
        else:
            return {
                "messages": messages,
                "model": SUPPORTED_INSTRUCT_MODELS.get(request.model, request.model),
                **options
            }

    async def embeddings(
        self,
        model: str,
        contents: List[InterleavedTextMedia],
    ) -> EmbeddingsResponse:
        raise NotImplementedError()
