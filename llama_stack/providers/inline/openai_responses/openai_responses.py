# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import AsyncIterator, List, Optional, cast

from llama_stack.apis.inference.inference import (
    Inference,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChoice,
    OpenAIMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.apis.models.models import Models, ModelType
from llama_stack.apis.openai_responses import OpenAIResponses
from llama_stack.apis.openai_responses.openai_responses import (
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseOutputMessage,
    OpenAIResponseOutputMessageContentOutputText,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.kvstore import kvstore_impl

from .config import OpenAIResponsesImplConfig

logger = get_logger(name=__name__, category="openai_responses")

OPENAI_RESPONSES_PREFIX = "openai_responses:"


async def _previous_response_to_messages(previous_response: OpenAIResponseObject) -> List[OpenAIMessageParam]:
    messages: List[OpenAIMessageParam] = []
    for output_message in previous_response.output:
        messages.append(OpenAIAssistantMessageParam(content=output_message.content[0].text))
    return messages


async def _openai_choices_to_output_messages(choices: List[OpenAIChoice]) -> List[OpenAIResponseOutputMessage]:
    output_messages = []
    for choice in choices:
        output_content = ""
        if isinstance(choice.message.content, str):
            output_content = choice.message.content
        elif isinstance(choice.message.content, OpenAIChatCompletionContentPartTextParam):
            output_content = choice.message.content.text
        # TODO: handle image content
        output_messages.append(
            OpenAIResponseOutputMessage(
                id=f"msg_{uuid.uuid4()}",
                content=[OpenAIResponseOutputMessageContentOutputText(text=output_content)],
                status="completed",
            )
        )
    return output_messages


class OpenAIResponsesImpl(OpenAIResponses):
    def __init__(self, config: OpenAIResponsesImplConfig, models_api: Models, inference_api: Inference):
        self.config = config
        self.models_api = models_api
        self.inference_api = inference_api

    async def initialize(self) -> None:
        self.kvstore = await kvstore_impl(self.config.kvstore)

    async def shutdown(self) -> None:
        logger.debug("OpenAIResponsesImpl.shutdown")
        pass

    async def get_openai_response(
        self,
        id: str,
    ) -> OpenAIResponseObject:
        key = f"{OPENAI_RESPONSES_PREFIX}{id}"
        response_json = await self.kvstore.get(key=key)
        if response_json is None:
            raise ValueError(f"OpenAI response with id '{id}' not found")
        return OpenAIResponseObject.model_validate_json(response_json)

    async def create_openai_response(
        self,
        input: str,
        model: str,
        previous_response_id: Optional[str] = None,
        store: Optional[bool] = True,
        stream: Optional[bool] = False,
    ):
        model_obj = await self.models_api.get_model(model)
        if model_obj is None:
            raise ValueError(f"Model '{model}' not found")
        if model_obj.model_type == ModelType.embedding:
            raise ValueError(f"Model '{model}' is an embedding model and does not support chat completions")

        messages: List[OpenAIMessageParam] = []
        if previous_response_id:
            previous_response = await self.get_openai_response(previous_response_id)
            messages.extend(await _previous_response_to_messages(previous_response))
        messages.append(OpenAIUserMessageParam(content=input))

        chat_response = await self.inference_api.openai_chat_completion(
            model=model_obj.identifier,
            messages=messages,
        )
        # type cast to appease mypy
        chat_response = cast(OpenAIChatCompletion, chat_response)

        output_messages = await _openai_choices_to_output_messages(chat_response.choices)
        response = OpenAIResponseObject(
            created_at=chat_response.created,
            id=f"resp-{uuid.uuid4()}",
            model=model_obj.identifier,
            object="response",
            status="completed",
            output=output_messages,
        )

        if store:
            # Store in kvstore
            key = f"{OPENAI_RESPONSES_PREFIX}{response.id}"
            await self.kvstore.set(
                key=key,
                value=response.model_dump_json(),
            )

        if stream:

            async def async_response() -> AsyncIterator[OpenAIResponseObjectStream]:
                yield OpenAIResponseObjectStream(response=response)

            return async_response()

        return response
