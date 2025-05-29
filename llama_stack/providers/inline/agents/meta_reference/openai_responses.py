# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from llama_stack.apis.agents import Order
from llama_stack.apis.agents.openai_responses import (
    AllowedToolsFilter,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContent,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolMCP,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseObjectStreamResponseCompleted,
    OpenAIResponseObjectStreamResponseCreated,
    OpenAIResponseObjectStreamResponseOutputTextDelta,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessageContent,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPListTools,
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
from llama_stack.apis.tools.tools import ToolGroups, ToolRuntime
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition
from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool
from llama_stack.providers.utils.responses.responses_store import ResponsesStore
from llama_stack.providers.utils.tools.mcp import invoke_mcp_tool, list_mcp_tools

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
    input_items: ListOpenAIResponseInputItem
    response: OpenAIResponseObject


class ChatCompletionContext(BaseModel):
    model: str
    messages: list[OpenAIMessageParam]
    tools: list[ChatCompletionToolParam] | None = None
    mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP]
    stream: bool
    temperature: float | None


class OpenAIResponsesImpl:
    def __init__(
        self,
        inference_api: Inference,
        tool_groups_api: ToolGroups,
        tool_runtime_api: ToolRuntime,
        responses_store: ResponsesStore,
    ):
        self.inference_api = inference_api
        self.tool_groups_api = tool_groups_api
        self.tool_runtime_api = tool_runtime_api
        self.responses_store = responses_store

    async def _prepend_previous_response(
        self, input: str | list[OpenAIResponseInput], previous_response_id: str | None = None
    ):
        if previous_response_id:
            previous_response_with_input = await self.responses_store.get_response_object(previous_response_id)

            # previous response input items
            new_input_items = previous_response_with_input.input

            # previous response output items
            new_input_items.extend(previous_response_with_input.output)

            # new input items from the current request
            if isinstance(input, str):
                new_input_items.append(OpenAIResponseMessage(content=input, role="user"))
            else:
                new_input_items.extend(input)

            input = new_input_items

        return input

    async def _prepend_instructions(self, messages, instructions):
        if instructions:
            messages.insert(0, OpenAISystemMessageParam(content=instructions))

    async def get_openai_response(
        self,
        response_id: str,
    ) -> OpenAIResponseObject:
        response_with_input = await self.responses_store.get_response_object(response_id)
        return OpenAIResponseObject(**{k: v for k, v in response_with_input.model_dump().items() if k != "input"})

    async def list_openai_responses(
        self,
        after: str | None = None,
        limit: int | None = 50,
        model: str | None = None,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseObject:
        return await self.responses_store.list_responses(after, limit, model, order)

    async def list_openai_response_input_items(
        self,
        response_id: str,
        after: str | None = None,
        before: str | None = None,
        include: list[str] | None = None,
        limit: int | None = 20,
        order: Order | None = Order.desc,
    ) -> ListOpenAIResponseInputItem:
        """List input items for a given OpenAI response.

        :param response_id: The ID of the response to retrieve input items for.
        :param after: An item ID to list items after, used for pagination.
        :param before: An item ID to list items before, used for pagination.
        :param include: Additional fields to include in the response.
        :param limit: A limit on the number of objects to be returned.
        :param order: The order to return the input items in.
        :returns: An ListOpenAIResponseInputItem.
        """
        return await self.responses_store.list_response_input_items(response_id, after, before, include, limit, order)

    async def _process_response_choices(
        self,
        chat_response: OpenAIChatCompletion,
        ctx: ChatCompletionContext,
        tools: list[OpenAIResponseInputTool] | None,
    ) -> list[OpenAIResponseOutput]:
        """Handle tool execution and response message creation."""
        output_messages: list[OpenAIResponseOutput] = []
        # Execute tool calls if any
        for choice in chat_response.choices:
            if choice.message.tool_calls and tools:
                # Assume if the first tool is a function, all tools are functions
                if tools[0].type == "function":
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
                    tool_messages = await self._execute_tool_and_return_final_output(choice, ctx)
                    output_messages.extend(tool_messages)
            else:
                output_messages.append(await _convert_chat_choice_to_response_message(choice))

        return output_messages

    async def _store_response(
        self,
        response: OpenAIResponseObject,
        input: str | list[OpenAIResponseInput],
    ) -> None:
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

        await self.responses_store.store_response_object(
            response_object=response,
            input=input_items_data,
        )

    async def create_openai_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        store: bool | None = True,
        stream: bool | None = False,
        temperature: float | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
    ):
        stream = False if stream is None else stream

        output_messages: list[OpenAIResponseOutput] = []

        # Input preprocessing
        input = await self._prepend_previous_response(input, previous_response_id)
        messages = await _convert_response_input_to_chat_messages(input)
        await self._prepend_instructions(messages, instructions)

        # Tool setup
        chat_tools, mcp_tool_to_server, mcp_list_message = (
            await self._convert_response_tools_to_chat_tools(tools) if tools else (None, {}, None)
        )
        if mcp_list_message:
            output_messages.append(mcp_list_message)

        ctx = ChatCompletionContext(
            model=model,
            messages=messages,
            tools=chat_tools,
            mcp_tool_to_server=mcp_tool_to_server,
            stream=stream,
            temperature=temperature,
        )

        inference_result = await self.inference_api.openai_chat_completion(
            model=model,
            messages=messages,
            tools=chat_tools,
            stream=stream,
            temperature=temperature,
        )

        if stream:
            return self._create_streaming_response(
                inference_result=inference_result,
                ctx=ctx,
                output_messages=output_messages,
                input=input,
                model=model,
                store=store,
                tools=tools,
            )
        else:
            return await self._create_non_streaming_response(
                inference_result=inference_result,
                ctx=ctx,
                output_messages=output_messages,
                input=input,
                model=model,
                store=store,
                tools=tools,
            )

    async def _create_non_streaming_response(
        self,
        inference_result: Any,
        ctx: ChatCompletionContext,
        output_messages: list[OpenAIResponseOutput],
        input: str | list[OpenAIResponseInput],
        model: str,
        store: bool | None,
        tools: list[OpenAIResponseInputTool] | None,
    ) -> OpenAIResponseObject:
        chat_response = OpenAIChatCompletion(**inference_result.model_dump())

        # Process response choices (tool execution and message creation)
        output_messages.extend(
            await self._process_response_choices(
                chat_response=chat_response,
                ctx=ctx,
                tools=tools,
            )
        )

        response = OpenAIResponseObject(
            created_at=chat_response.created,
            id=f"resp-{uuid.uuid4()}",
            model=model,
            object="response",
            status="completed",
            output=output_messages,
        )
        logger.debug(f"OpenAI Responses response: {response}")

        # Store response if requested
        if store:
            await self._store_response(
                response=response,
                input=input,
            )

        return response

    async def _create_streaming_response(
        self,
        inference_result: Any,
        ctx: ChatCompletionContext,
        output_messages: list[OpenAIResponseOutput],
        input: str | list[OpenAIResponseInput],
        model: str,
        store: bool | None,
        tools: list[OpenAIResponseInputTool] | None,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        # Create initial response and emit response.created immediately
        response_id = f"resp-{uuid.uuid4()}"
        created_at = int(time.time())

        initial_response = OpenAIResponseObject(
            created_at=created_at,
            id=response_id,
            model=model,
            object="response",
            status="in_progress",
            output=output_messages.copy(),
        )

        # Emit response.created immediately
        yield OpenAIResponseObjectStreamResponseCreated(response=initial_response)

        # For streaming, inference_result is an async iterator of chunks
        # Stream chunks and emit delta events as they arrive
        chat_response_id = ""
        chat_response_content = []
        chat_response_tool_calls: dict[int, OpenAIChatCompletionToolCall] = {}
        chunk_created = 0
        chunk_model = ""
        chunk_finish_reason = ""
        sequence_number = 0

        # Create a placeholder message item for delta events
        message_item_id = f"msg_{uuid.uuid4()}"

        async for chunk in inference_result:
            chat_response_id = chunk.id
            chunk_created = chunk.created
            chunk_model = chunk.model
            for chunk_choice in chunk.choices:
                # Emit incremental text content as delta events
                if chunk_choice.delta.content:
                    sequence_number += 1
                    yield OpenAIResponseObjectStreamResponseOutputTextDelta(
                        content_index=0,
                        delta=chunk_choice.delta.content,
                        item_id=message_item_id,
                        output_index=0,
                        sequence_number=sequence_number,
                    )

                # Collect content for final response
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
                            tool_call_dict.pop("type", None)
                            response_tool_call = OpenAIChatCompletionToolCall(**tool_call_dict)
                        chat_response_tool_calls[tool_call.index] = response_tool_call

        # Convert collected chunks to complete response
        if chat_response_tool_calls:
            tool_calls = [chat_response_tool_calls[i] for i in sorted(chat_response_tool_calls.keys())]
        else:
            tool_calls = None
        assistant_message = OpenAIAssistantMessageParam(
            content="".join(chat_response_content),
            tool_calls=tool_calls,
        )
        chat_response_obj = OpenAIChatCompletion(
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

        # Process response choices (tool execution and message creation)
        output_messages.extend(
            await self._process_response_choices(
                chat_response=chat_response_obj,
                ctx=ctx,
                tools=tools,
            )
        )

        # Create final response
        final_response = OpenAIResponseObject(
            created_at=created_at,
            id=response_id,
            model=model,
            object="response",
            status="completed",
            output=output_messages,
        )

        if store:
            await self._store_response(
                response=final_response,
                input=input,
            )

        # Emit response.completed
        yield OpenAIResponseObjectStreamResponseCompleted(response=final_response)

    async def _convert_response_tools_to_chat_tools(
        self, tools: list[OpenAIResponseInputTool]
    ) -> tuple[
        list[ChatCompletionToolParam],
        dict[str, OpenAIResponseInputToolMCP],
        OpenAIResponseOutput | None,
    ]:
        from llama_stack.apis.agents.openai_responses import (
            MCPListToolsTool,
        )
        from llama_stack.apis.tools.tools import Tool

        mcp_tool_to_server = {}

        def make_openai_tool(tool_name: str, tool: Tool) -> ChatCompletionToolParam:
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
            return convert_tooldef_to_openai_tool(tool_def)

        mcp_list_message = None
        chat_tools: list[ChatCompletionToolParam] = []
        for input_tool in tools:
            # TODO: Handle other tool types
            if input_tool.type == "function":
                chat_tools.append(ChatCompletionToolParam(type="function", function=input_tool.model_dump()))
            elif input_tool.type == "web_search":
                tool_name = "web_search"
                tool = await self.tool_groups_api.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")
                chat_tools.append(make_openai_tool(tool_name, tool))
            elif input_tool.type == "mcp":
                always_allowed = None
                never_allowed = None
                if input_tool.allowed_tools:
                    if isinstance(input_tool.allowed_tools, list):
                        always_allowed = input_tool.allowed_tools
                    elif isinstance(input_tool.allowed_tools, AllowedToolsFilter):
                        always_allowed = input_tool.allowed_tools.always
                        never_allowed = input_tool.allowed_tools.never

                tool_defs = await list_mcp_tools(
                    endpoint=input_tool.server_url,
                    headers=input_tool.headers or {},
                )

                mcp_list_message = OpenAIResponseOutputMessageMCPListTools(
                    id=f"mcp_list_{uuid.uuid4()}",
                    status="completed",
                    server_label=input_tool.server_label,
                    tools=[],
                )
                for t in tool_defs.data:
                    if never_allowed and t.name in never_allowed:
                        continue
                    if not always_allowed or t.name in always_allowed:
                        chat_tools.append(make_openai_tool(t.name, t))
                        if t.name in mcp_tool_to_server:
                            raise ValueError(f"Duplicate tool name {t.name} found for server {input_tool.server_label}")
                        mcp_tool_to_server[t.name] = input_tool
                        mcp_list_message.tools.append(
                            MCPListToolsTool(
                                name=t.name,
                                description=t.description,
                                input_schema={
                                    "type": "object",
                                    "properties": {
                                        p.name: {
                                            "type": p.parameter_type,
                                            "description": p.description,
                                        }
                                        for p in t.parameters
                                    },
                                    "required": [p.name for p in t.parameters if p.required],
                                },
                            )
                        )
            else:
                raise ValueError(f"Llama Stack OpenAI Responses does not yet support tool type: {input_tool.type}")
        return chat_tools, mcp_tool_to_server, mcp_list_message

    async def _execute_tool_and_return_final_output(
        self,
        choice: OpenAIChoice,
        ctx: ChatCompletionContext,
    ) -> list[OpenAIResponseOutput]:
        output_messages: list[OpenAIResponseOutput] = []

        if not isinstance(choice.message, OpenAIAssistantMessageParam):
            return output_messages

        if not choice.message.tool_calls:
            return output_messages

        next_turn_messages = ctx.messages.copy()

        # Add the assistant message with tool_calls response to the messages list
        next_turn_messages.append(choice.message)

        for tool_call in choice.message.tool_calls:
            # TODO: telemetry spans for tool calls
            tool_call_log, further_input = await self._execute_tool_call(tool_call, ctx)
            if tool_call_log:
                output_messages.append(tool_call_log)
            if further_input:
                next_turn_messages.append(further_input)

        tool_results_chat_response = await self.inference_api.openai_chat_completion(
            model=ctx.model,
            messages=next_turn_messages,
            stream=ctx.stream,
            temperature=ctx.temperature,
        )
        # type cast to appease mypy: this is needed because we don't handle streaming properly :)
        tool_results_chat_response = cast(OpenAIChatCompletion, tool_results_chat_response)

        # Huge TODO: these are NOT the final outputs, we must keep the loop going
        tool_final_outputs = [
            await _convert_chat_choice_to_response_message(choice) for choice in tool_results_chat_response.choices
        ]
        # TODO: Wire in annotations with URLs, titles, etc to these output messages
        output_messages.extend(tool_final_outputs)
        return output_messages

    async def _execute_tool_call(
        self,
        tool_call: OpenAIChatCompletionToolCall,
        ctx: ChatCompletionContext,
    ) -> tuple[OpenAIResponseOutput | None, OpenAIMessageParam | None]:
        from llama_stack.providers.utils.inference.prompt_adapter import (
            interleaved_content_as_str,
        )

        tool_call_id = tool_call.id
        function = tool_call.function

        if not function or not tool_call_id or not function.name:
            return None, None

        error_exc = None
        result = None
        try:
            if function.name in ctx.mcp_tool_to_server:
                mcp_tool = ctx.mcp_tool_to_server[function.name]
                result = await invoke_mcp_tool(
                    endpoint=mcp_tool.server_url,
                    headers=mcp_tool.headers or {},
                    tool_name=function.name,
                    kwargs=json.loads(function.arguments) if function.arguments else {},
                )
            else:
                result = await self.tool_runtime_api.invoke_tool(
                    tool_name=function.name,
                    kwargs=json.loads(function.arguments) if function.arguments else {},
                )
        except Exception as e:
            error_exc = e

        if function.name in ctx.mcp_tool_to_server:
            from llama_stack.apis.agents.openai_responses import OpenAIResponseOutputMessageMCPCall

            message = OpenAIResponseOutputMessageMCPCall(
                id=tool_call_id,
                arguments=function.arguments,
                name=function.name,
                server_label=ctx.mcp_tool_to_server[function.name].server_label,
            )
            if error_exc:
                message.error = str(error_exc)
            elif (result.error_code and result.error_code > 0) or result.error_message:
                message.error = f"Error (code {result.error_code}): {result.error_message}"
            elif result.content:
                message.output = interleaved_content_as_str(result.content)
        else:
            if function.name == "web_search":
                message = OpenAIResponseOutputMessageWebSearchToolCall(
                    id=tool_call_id,
                    status="completed",
                )
                if error_exc or (result.error_code and result.error_code > 0) or result.error_message:
                    message.status = "failed"
            else:
                raise ValueError(f"Unknown tool {function.name} called")

        input_message = None
        if result and result.content:
            if isinstance(result.content, str):
                content = result.content
            elif isinstance(result.content, list):
                from llama_stack.apis.common.content_types import ImageContentItem, TextContentItem

                content = []
                for item in result.content:
                    if isinstance(item, TextContentItem):
                        part = OpenAIChatCompletionContentPartTextParam(text=item.text)
                    elif isinstance(item, ImageContentItem):
                        if item.image.data:
                            url = f"data:image;base64,{item.image.data}"
                        else:
                            url = item.image.url
                        part = OpenAIChatCompletionContentPartImageParam(image_url=OpenAIImageURL(url=url))
                    else:
                        raise ValueError(f"Unknown result content type: {type(item)}")
                    content.append(part)
            else:
                raise ValueError(f"Unknown result content type: {type(result.content)}")
            input_message = OpenAIToolMessageParam(content=content, tool_call_id=tool_call_id)

        return message, input_message
