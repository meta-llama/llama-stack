# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import uuid
from collections.abc import AsyncIterator
from typing import cast

from openai.types.chat import ChatCompletionToolParam

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputItemList,
    OpenAIResponseInputMessageContent,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageWebSearchToolCall,
    OpenAIResponsePreviousResponseWithInputItems,
)
from llama_stack.apis.inference.inference import (
    Inference,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIImageURL,
    OpenAIMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.apis.tools.tools import ToolGroups, ToolInvocationResult, ToolRuntime
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition
from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool
from llama_stack.providers.utils.kvstore import KVStore

logger = get_logger(name=__name__, category="openai_responses")

OPENAI_RESPONSES_PREFIX = "openai_responses:"


async def _convert_response_input_content_to_chat_content_parts(
    input_content: list[OpenAIResponseInputMessageContent],
) -> list[OpenAIChatCompletionContentPartParam]:
    """
    Convert a list of input content items to a list of chat completion content parts
    """
    content_parts = []
    for input_content_part in input_content:
        if isinstance(input_content_part, OpenAIResponseInputMessageContentText):
            content_parts.append(OpenAIChatCompletionContentPartTextParam(text=input_content_part.text))
        elif isinstance(input_content_part, OpenAIResponseInputMessageContentImage):
            if input_content_part.image_url:
                image_url = OpenAIImageURL(url=input_content_part.image_url, detail=input_content_part.detail)
                content_parts.append(OpenAIChatCompletionContentPartImageParam(image_url=image_url))
    return content_parts


async def _convert_response_input_to_chat_user_content(
    input: str | list[OpenAIResponseInput],
) -> str | list[OpenAIChatCompletionContentPartParam]:
    user_content: str | list[OpenAIChatCompletionContentPartParam] = ""
    if isinstance(input, list):
        user_content = []
        for user_input in input:
            if isinstance(user_input.content, list):
                user_content.extend(await _convert_response_input_content_to_chat_content_parts(user_input.content))
            else:
                user_content.append(OpenAIChatCompletionContentPartTextParam(text=user_input.content))
    else:
        user_content = input
    return user_content


async def _openai_choices_to_output_messages(choices: list[OpenAIChoice]) -> list[OpenAIResponseMessage]:
    output_messages = []
    for choice in choices:
        output_content = ""
        if isinstance(choice.message.content, str):
            output_content = choice.message.content
        elif isinstance(choice.message.content, OpenAIChatCompletionContentPartTextParam):
            output_content = choice.message.content.text
        # TODO: handle image content
        output_messages.append(
            OpenAIResponseMessage(
                id=f"msg_{uuid.uuid4()}",
                content=[OpenAIResponseOutputMessageContentOutputText(text=output_content)],
                status="completed",
                role="assistant",
            )
        )
    return output_messages


class OpenAIResponsesImpl:
    def __init__(
        self,
        persistence_store: KVStore,
        inference_api: Inference,
        tool_groups_api: ToolGroups,
        tool_runtime_api: ToolRuntime,
    ):
        self.persistence_store = persistence_store
        self.inference_api = inference_api
        self.tool_groups_api = tool_groups_api
        self.tool_runtime_api = tool_runtime_api

    async def _get_previous_response_with_input(self, id: str) -> OpenAIResponsePreviousResponseWithInputItems:
        key = f"{OPENAI_RESPONSES_PREFIX}{id}"
        response_json = await self.persistence_store.get(key=key)
        if response_json is None:
            raise ValueError(f"OpenAI response with id '{id}' not found")
        return OpenAIResponsePreviousResponseWithInputItems.model_validate_json(response_json)

    async def _prepend_previous_response(
        self, input: str | list[OpenAIResponseInput], previous_response_id: str | None = None
    ):
        if previous_response_id:
            previous_response_with_input = await self._get_previous_response_with_input(previous_response_id)

            # previous response input items
            new_input_items = previous_response_with_input.input_items.data

            # previous response output items
            new_input_items.extend(previous_response_with_input.response.output)

            # new input items from the current request
            if isinstance(input, str):
                # Normalize input to a list of OpenAIResponseInputMessage objects
                input = [OpenAIResponseMessage(content=input, role="user")]
            new_input_items.extend(input)
            input = new_input_items

        return input

    async def get_openai_response(
        self,
        id: str,
    ) -> OpenAIResponseObject:
        response_with_input = await self._get_previous_response_with_input(id)
        return response_with_input.response

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        previous_response_id: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
    ):
        stream = False if stream is None else stream

        input = await self._prepend_previous_response(input, previous_response_id)

        messages: list[OpenAIMessageParam] = []

        user_content = await _convert_response_input_to_chat_user_content(input)
        messages.append(OpenAIUserMessageParam(content=user_content))

        chat_tools = await self._convert_response_tools_to_chat_tools(tools) if tools else None
        chat_response = await self.inference_api.openai_chat_completion(
            model=model,
            messages=messages,
            tools=chat_tools,
            stream=stream,
            temperature=temperature,
        )

        if stream:
            # TODO: refactor this into a separate method that handles streaming
            chat_response_id = ""
            chat_response_content = []
            # TODO: these chunk_ fields are hacky and only take the last chunk into account
            chunk_created = 0
            chunk_model = ""
            chunk_finish_reason = ""
            async for chunk in chat_response:
                chat_response_id = chunk.id
                chunk_created = chunk.created
                chunk_model = chunk.model
                for chunk_choice in chunk.choices:
                    # TODO: this only works for text content
                    chat_response_content.append(chunk_choice.delta.content or "")
                    if chunk_choice.finish_reason:
                        chunk_finish_reason = chunk_choice.finish_reason
            assistant_message = OpenAIAssistantMessageParam(content="".join(chat_response_content))
            chat_response = OpenAIChatCompletion(
                id=chat_response_id,
                choices=[
                    OpenAIChoice(
                        message=assistant_message,
                        finish_reason=chunk_finish_reason,
                        index=0,
                    )
                ],
                created=chunk_created,
                model=chunk_model,
            )
        else:
            # dump and reload to map to our pydantic types
            chat_response = OpenAIChatCompletion(**chat_response.model_dump())

        output_messages: list[OpenAIResponseOutput] = []
        if chat_response.choices[0].message.tool_calls:
            output_messages.extend(
                await self._execute_tool_and_return_final_output(model, stream, chat_response, messages, temperature)
            )
        else:
            output_messages.extend(await _openai_choices_to_output_messages(chat_response.choices))
        response = OpenAIResponseObject(
            created_at=chat_response.created,
            id=f"resp-{uuid.uuid4()}",
            model=model,
            object="response",
            status="completed",
            output=output_messages,
        )

        if store:
            # Store in kvstore

            new_input_id = f"msg_{uuid.uuid4()}"
            if isinstance(input, str):
                # synthesize a message from the input string
                input_content = OpenAIResponseInputMessageContentText(text=input)
                input_content_item = OpenAIResponseMessage(
                    role="user",
                    content=[input_content],
                    id=new_input_id,
                )
                input_items_data = [input_content_item]
            else:
                # we already have a list of messages
                input_items_data = []
                for input_item in input:
                    if isinstance(input_item, OpenAIResponseMessage):
                        # These may or may not already have an id, so dump to dict, check for id, and add if missing
                        input_item_dict = input_item.model_dump()
                        if "id" not in input_item_dict:
                            input_item_dict["id"] = new_input_id
                        input_items_data.append(OpenAIResponseMessage(**input_item_dict))
                    else:
                        input_items_data.append(input_item)

            input_items = OpenAIResponseInputItemList(data=input_items_data)
            prev_response = OpenAIResponsePreviousResponseWithInputItems(
                input_items=input_items,
                response=response,
            )
            key = f"{OPENAI_RESPONSES_PREFIX}{response.id}"
            await self.persistence_store.set(
                key=key,
                value=prev_response.model_dump_json(),
            )

        if stream:

            async def async_response() -> AsyncIterator[OpenAIResponseObjectStream]:
                # TODO: response created should actually get emitted much earlier in the process
                yield OpenAIResponseObjectStreamResponseCreated(response=response)
                yield OpenAIResponseObjectStreamResponseCompleted(response=response)

            return async_response()

        return response

    async def _convert_response_tools_to_chat_tools(
        self, tools: list[OpenAIResponseInputTool]
    ) -> list[ChatCompletionToolParam]:
        chat_tools: list[ChatCompletionToolParam] = []
        for input_tool in tools:
            # TODO: Handle other tool types
            if input_tool.type == "web_search":
                tool_name = "web_search"
                tool = await self.tool_groups_api.get_tool(tool_name)
                tool_def = ToolDefinition(
                    tool_name=tool_name,
                    description=tool.description,
                    parameters={
                        param.name: ToolParamDefinition(
                            param_type=param.parameter_type,
                            description=param.description,
                            required=param.required,
                            default=param.default,
                        )
                        for param in tool.parameters
                    },
                )
                chat_tool = convert_tooldef_to_openai_tool(tool_def)
                chat_tools.append(chat_tool)
            else:
                raise ValueError(f"Llama Stack OpenAI Responses does not yet support tool type: {input_tool.type}")
        return chat_tools

    async def _execute_tool_and_return_final_output(
        self,
        model_id: str,
        stream: bool,
        chat_response: OpenAIChatCompletion,
        messages: list[OpenAIMessageParam],
        temperature: float,
    ) -> list[OpenAIResponseOutput]:
        output_messages: list[OpenAIResponseOutput] = []
        choice = chat_response.choices[0]

        # If the choice is not an assistant message, we don't need to execute any tools
        if not isinstance(choice.message, OpenAIAssistantMessageParam):
            return output_messages

        # If the assistant message doesn't have any tool calls, we don't need to execute any tools
        if not choice.message.tool_calls:
            return output_messages

        # Add the assistant message with tool_calls response to the messages list
        messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            tool_call_id = tool_call.id
            function = tool_call.function

            # If for some reason the tool call doesn't have a function or id, we can't execute it
            if not function or not tool_call_id:
                continue

            # TODO: telemetry spans for tool calls
            result = await self._execute_tool_call(function)

            # Handle tool call failure
            if not result:
                output_messages.append(
                    OpenAIResponseOutputMessageWebSearchToolCall(
                        id=tool_call_id,
                        status="failed",
                    )
                )
                continue

            output_messages.append(
                OpenAIResponseOutputMessageWebSearchToolCall(
                    id=tool_call_id,
                    status="completed",
                ),
            )

            result_content = ""
            # TODO: handle other result content types and lists
            if isinstance(result.content, str):
                result_content = result.content
            messages.append(OpenAIToolMessageParam(content=result_content, tool_call_id=tool_call_id))

        tool_results_chat_response = await self.inference_api.openai_chat_completion(
            model=model_id,
            messages=messages,
            stream=stream,
            temperature=temperature,
        )
        # type cast to appease mypy
        tool_results_chat_response = cast(OpenAIChatCompletion, tool_results_chat_response)
        tool_final_outputs = await _openai_choices_to_output_messages(tool_results_chat_response.choices)
        # TODO: Wire in annotations with URLs, titles, etc to these output messages
        output_messages.extend(tool_final_outputs)
        return output_messages

    async def _execute_tool_call(
        self,
        function: OpenAIChatCompletionToolCallFunction,
    ) -> ToolInvocationResult | None:
        if not function.name:
            return None
        function_args = json.loads(function.arguments) if function.arguments else {}
        logger.info(f"executing tool call: {function.name} with args: {function_args}")
        result = await self.tool_runtime_api.invoke_tool(
            tool_name=function.name,
            kwargs=function_args,
        )
        logger.debug(f"tool call {function.name} completed with result: {result}")
        return result
