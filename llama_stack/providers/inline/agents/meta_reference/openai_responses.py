# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from llama_stack.apis.agents.openai_responses import (
    OpenAIResponseInput,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputItemList,
    OpenAIResponseInputMessageContent,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolFunction,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessageContent,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageWebSearchToolCall,
)
from llama_stack.apis.inference.inference import (
    Inference,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionContentPartImageParam,
    OpenAIChatCompletionContentPartParam,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionToolCall,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIDeveloperMessageParam,
    OpenAIImageURL,
    OpenAIMessageParam,
    OpenAISystemMessageParam,
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


async def _convert_response_content_to_chat_content(
    content: str | list[OpenAIResponseInputMessageContent] | list[OpenAIResponseOutputMessageContent],
) -> str | list[OpenAIChatCompletionContentPartParam]:
    """
    Convert the content parts from an OpenAI Response API request into OpenAI Chat Completion content parts.

    The content schemas of each API look similar, but are not exactly the same.
    """
    if isinstance(content, str):
        return content

    converted_parts = []
    for content_part in content:
        if isinstance(content_part, OpenAIResponseInputMessageContentText):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part.text))
        elif isinstance(content_part, OpenAIResponseOutputMessageContentOutputText):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part.text))
        elif isinstance(content_part, OpenAIResponseInputMessageContentImage):
            if content_part.image_url:
                image_url = OpenAIImageURL(url=content_part.image_url, detail=content_part.detail)
                converted_parts.append(OpenAIChatCompletionContentPartImageParam(image_url=image_url))
        elif isinstance(content_part, str):
            converted_parts.append(OpenAIChatCompletionContentPartTextParam(text=content_part))
        else:
            raise ValueError(
                f"Llama Stack OpenAI Responses does not yet support content type '{type(content_part)}' in this context"
            )
    return converted_parts


async def _convert_response_input_to_chat_messages(
    input: str | list[OpenAIResponseInput],
) -> list[OpenAIMessageParam]:
    """
    Convert the input from an OpenAI Response API request into OpenAI Chat Completion messages.
    """
    messages: list[OpenAIMessageParam] = []
    if isinstance(input, list):
        for input_item in input:
            if isinstance(input_item, OpenAIResponseInputFunctionToolCallOutput):
                messages.append(
                    OpenAIToolMessageParam(
                        content=input_item.output,
                        tool_call_id=input_item.call_id,
                    )
                )
            elif isinstance(input_item, OpenAIResponseOutputMessageFunctionToolCall):
                tool_call = OpenAIChatCompletionToolCall(
                    index=0,
                    id=input_item.call_id,
                    function=OpenAIChatCompletionToolCallFunction(
                        name=input_item.name,
                        arguments=input_item.arguments,
                    ),
                )
                messages.append(OpenAIAssistantMessageParam(tool_calls=[tool_call]))
            else:
                content = await _convert_response_content_to_chat_content(input_item.content)
                message_type = await _get_message_type_by_role(input_item.role)
                if message_type is None:
                    raise ValueError(
                        f"Llama Stack OpenAI Responses does not yet support message role '{input_item.role}' in this context"
                    )
                messages.append(message_type(content=content))
    else:
        messages.append(OpenAIUserMessageParam(content=input))
    return messages


async def _convert_chat_choice_to_response_message(choice: OpenAIChoice) -> OpenAIResponseMessage:
    """
    Convert an OpenAI Chat Completion choice into an OpenAI Response output message.
    """
    output_content = ""
    if isinstance(choice.message.content, str):
        output_content = choice.message.content
    elif isinstance(choice.message.content, OpenAIChatCompletionContentPartTextParam):
        output_content = choice.message.content.text
    else:
        raise ValueError(
            f"Llama Stack OpenAI Responses does not yet support output content type: {type(choice.message.content)}"
        )

    return OpenAIResponseMessage(
        id=f"msg_{uuid.uuid4()}",
        content=[OpenAIResponseOutputMessageContentOutputText(text=output_content)],
        status="completed",
        role="assistant",
    )


async def _get_message_type_by_role(role: str):
    role_to_type = {
        "user": OpenAIUserMessageParam,
        "system": OpenAISystemMessageParam,
        "assistant": OpenAIAssistantMessageParam,
        "developer": OpenAIDeveloperMessageParam,
    }
    return role_to_type.get(role)


class OpenAIResponsePreviousResponseWithInputItems(BaseModel):
    input_items: OpenAIResponseInputItemList
    response: OpenAIResponseObject


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
                new_input_items.append(OpenAIResponseMessage(content=input, role="user"))
            else:
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
        messages = await _convert_response_input_to_chat_messages(input)
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
            chat_response_tool_calls: dict[int, OpenAIChatCompletionToolCall] = {}
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

                    # Aggregate tool call arguments across chunks, using their index as the aggregation key
                    if chunk_choice.delta.tool_calls:
                        for tool_call in chunk_choice.delta.tool_calls:
                            response_tool_call = chat_response_tool_calls.get(tool_call.index, None)
                            if response_tool_call:
                                response_tool_call.function.arguments += tool_call.function.arguments
                            else:
                                tool_call_dict: dict[str, Any] = tool_call.model_dump()
                                # Ensure we don't have any empty type field in the tool call dict.
                                # The OpenAI client used by providers often returns a type=None here.
                                tool_call_dict.pop("type", None)
                                response_tool_call = OpenAIChatCompletionToolCall(**tool_call_dict)
                            chat_response_tool_calls[tool_call.index] = response_tool_call

            # Convert the dict of tool calls by index to a list of tool calls to pass back in our response
            if chat_response_tool_calls:
                tool_calls = [chat_response_tool_calls[i] for i in sorted(chat_response_tool_calls.keys())]
            else:
                tool_calls = None
            assistant_message = OpenAIAssistantMessageParam(
                content="".join(chat_response_content),
                tool_calls=tool_calls,
            )
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
        for choice in chat_response.choices:
            if choice.message.tool_calls and tools:
                # Assume if the first tool is a function, all tools are functions
                if isinstance(tools[0], OpenAIResponseInputToolFunction):
                    for tool_call in choice.message.tool_calls:
                        output_messages.append(
                            OpenAIResponseOutputMessageFunctionToolCall(
                                arguments=tool_call.function.arguments or "",
                                call_id=tool_call.id,
                                name=tool_call.function.name or "",
                                id=f"fc_{uuid.uuid4()}",
                                status="completed",
                            )
                        )
                else:
                    output_messages.extend(
                        await self._execute_tool_and_return_final_output(model, stream, choice, messages, temperature)
                    )
            else:
                output_messages.append(await _convert_chat_choice_to_response_message(choice))
        response = OpenAIResponseObject(
            created_at=chat_response.created,
            id=f"resp-{uuid.uuid4()}",
            model=model,
            object="response",
            status="completed",
            output=output_messages,
        )
        logger.debug(f"OpenAI Responses response: {response}")

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
            if input_tool.type == "function":
                chat_tools.append(ChatCompletionToolParam(type="function", function=input_tool.model_dump()))
            elif input_tool.type == "web_search":
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
        choice: OpenAIChoice,
        messages: list[OpenAIMessageParam],
        temperature: float,
    ) -> list[OpenAIResponseOutput]:
        output_messages: list[OpenAIResponseOutput] = []

        # If the choice is not an assistant message, we don't need to execute any tools
        if not isinstance(choice.message, OpenAIAssistantMessageParam):
            return output_messages

        # If the assistant message doesn't have any tool calls, we don't need to execute any tools
        if not choice.message.tool_calls:
            return output_messages

        # Copy the messages list to avoid mutating the original list
        messages = messages.copy()

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
        tool_final_outputs = [
            await _convert_chat_choice_to_response_message(choice) for choice in tool_results_chat_response.choices
        ]
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
