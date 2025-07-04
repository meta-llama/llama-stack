# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from llama_stack.apis.agents import Order
from llama_stack.apis.agents.openai_responses import (
    AllowedToolsFilter,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputFunctionToolCallOutput,
    OpenAIResponseInputMessageContent,
    OpenAIResponseInputMessageContentImage,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolFileSearch,
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
    OpenAIResponseOutputMessageFileSearchToolCall,
    OpenAIResponseOutputMessageFunctionToolCall,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseOutputMessageWebSearchToolCall,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
    WebSearchToolTypes,
)
from llama_stack.apis.common.content_types import TextContentItem
from llama_stack.apis.inference import (
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
    OpenAIJSONSchema,
    OpenAIMessageParam,
    OpenAIResponseFormatJSONObject,
    OpenAIResponseFormatJSONSchema,
    OpenAIResponseFormatParam,
    OpenAIResponseFormatText,
    OpenAISystemMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.apis.tools import ToolGroups, ToolInvocationResult, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition
from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool
from llama_stack.providers.utils.responses.responses_store import ResponsesStore

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


async def _convert_response_text_to_chat_response_format(text: OpenAIResponseText) -> OpenAIResponseFormatParam:
    """
    Convert an OpenAI Response text parameter into an OpenAI Chat Completion response format.
    """
    if not text.format or text.format["type"] == "text":
        return OpenAIResponseFormatText(type="text")
    if text.format["type"] == "json_object":
        return OpenAIResponseFormatJSONObject()
    if text.format["type"] == "json_schema":
        return OpenAIResponseFormatJSONSchema(
            json_schema=OpenAIJSONSchema(name=text.format["name"], schema=text.format["schema"])
        )
    raise ValueError(f"Unsupported text format: {text.format}")


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
    response_tools: list[OpenAIResponseInputTool] | None = None
    chat_tools: list[ChatCompletionToolParam] | None = None
    mcp_tool_to_server: dict[str, OpenAIResponseInputToolMCP]
    temperature: float | None
    response_format: OpenAIResponseFormatParam


class OpenAIResponsesImpl:
    def __init__(
        self,
        inference_api: Inference,
        tool_groups_api: ToolGroups,
        tool_runtime_api: ToolRuntime,
        responses_store: ResponsesStore,
        vector_io_api: VectorIO,  # VectorIO
    ):
        self.inference_api = inference_api
        self.tool_groups_api = tool_groups_api
        self.tool_runtime_api = tool_runtime_api
        self.responses_store = responses_store
        self.vector_io_api = vector_io_api

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
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        max_infer_iters: int | None = 10,
    ):
        stream = bool(stream)
        text = OpenAIResponseText(format=OpenAIResponseTextFormat(type="text")) if text is None else text

        stream_gen = self._create_streaming_response(
            input=input,
            model=model,
            instructions=instructions,
            previous_response_id=previous_response_id,
            store=store,
            temperature=temperature,
            text=text,
            tools=tools,
            max_infer_iters=max_infer_iters,
        )

        if stream:
            return stream_gen
        else:
            response = None
            async for stream_chunk in stream_gen:
                if stream_chunk.type == "response.completed":
                    if response is not None:
                        raise ValueError("The response stream completed multiple times! Earlier response: {response}")
                    response = stream_chunk.response
                    # don't leave the generator half complete!

            if response is None:
                raise ValueError("The response stream never completed")
            return response

    async def _create_streaming_response(
        self,
        input: str | list[OpenAIResponseInput],
        model: str,
        instructions: str | None = None,
        previous_response_id: str | None = None,
        store: bool | None = True,
        temperature: float | None = None,
        text: OpenAIResponseText | None = None,
        tools: list[OpenAIResponseInputTool] | None = None,
        max_infer_iters: int | None = 10,
    ) -> AsyncIterator[OpenAIResponseObjectStream]:
        output_messages: list[OpenAIResponseOutput] = []

        # Input preprocessing
        input = await self._prepend_previous_response(input, previous_response_id)
        messages = await _convert_response_input_to_chat_messages(input)
        await self._prepend_instructions(messages, instructions)

        # Structured outputs
        response_format = await _convert_response_text_to_chat_response_format(text)

        # Tool setup, TODO: refactor this slightly since this can also yield events
        chat_tools, mcp_tool_to_server, mcp_list_message = (
            await self._convert_response_tools_to_chat_tools(tools) if tools else (None, {}, None)
        )
        if mcp_list_message:
            output_messages.append(mcp_list_message)

        ctx = ChatCompletionContext(
            model=model,
            messages=messages,
            response_tools=tools,
            chat_tools=chat_tools,
            mcp_tool_to_server=mcp_tool_to_server,
            temperature=temperature,
            response_format=response_format,
        )

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
            text=text,
        )

        yield OpenAIResponseObjectStreamResponseCreated(response=initial_response)

        n_iter = 0
        messages = ctx.messages.copy()

        while True:
            completion_result = await self.inference_api.openai_chat_completion(
                model=ctx.model,
                messages=messages,
                tools=ctx.chat_tools,
                stream=True,
                temperature=ctx.temperature,
                response_format=ctx.response_format,
            )

            # Process streaming chunks and build complete response
            chat_response_id = ""
            chat_response_content = []
            chat_response_tool_calls: dict[int, OpenAIChatCompletionToolCall] = {}
            chunk_created = 0
            chunk_model = ""
            chunk_finish_reason = ""
            sequence_number = 0

            # Create a placeholder message item for delta events
            message_item_id = f"msg_{uuid.uuid4()}"

            async for chunk in completion_result:
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

                    # Aggregate tool call arguments across chunks
                    if chunk_choice.delta.tool_calls:
                        for tool_call in chunk_choice.delta.tool_calls:
                            response_tool_call = chat_response_tool_calls.get(tool_call.index, None)
                            if response_tool_call:
                                # Don't attempt to concatenate arguments if we don't have any new argumentsAdd commentMore actions
                                if tool_call.function.arguments:
                                    # Guard against an initial None argument before we concatenate
                                    response_tool_call.function.arguments = (
                                        response_tool_call.function.arguments or ""
                                    ) + tool_call.function.arguments
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
            current_response = OpenAIChatCompletion(
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

            function_tool_calls = []
            non_function_tool_calls = []

            next_turn_messages = messages.copy()
            for choice in current_response.choices:
                next_turn_messages.append(choice.message)

                if choice.message.tool_calls and tools:
                    for tool_call in choice.message.tool_calls:
                        if _is_function_tool_call(tool_call, tools):
                            function_tool_calls.append(tool_call)
                        else:
                            non_function_tool_calls.append(tool_call)
                else:
                    output_messages.append(await _convert_chat_choice_to_response_message(choice))

            # execute non-function tool calls
            for tool_call in non_function_tool_calls:
                tool_call_log, tool_response_message = await self._execute_tool_call(tool_call, ctx)
                if tool_call_log:
                    output_messages.append(tool_call_log)
                if tool_response_message:
                    next_turn_messages.append(tool_response_message)

            for tool_call in function_tool_calls:
                output_messages.append(
                    OpenAIResponseOutputMessageFunctionToolCall(
                        arguments=tool_call.function.arguments or "",
                        call_id=tool_call.id,
                        name=tool_call.function.name or "",
                        id=f"fc_{uuid.uuid4()}",
                        status="completed",
                    )
                )

            if not function_tool_calls and not non_function_tool_calls:
                break

            if function_tool_calls:
                logger.info("Exiting inference loop since there is a function (client-side) tool call")
                break

            n_iter += 1
            if n_iter >= max_infer_iters:
                logger.info(f"Exiting inference loop since iteration count({n_iter}) exceeds {max_infer_iters=}")
                break

            messages = next_turn_messages

        # Create final response
        final_response = OpenAIResponseObject(
            created_at=created_at,
            id=response_id,
            model=model,
            object="response",
            status="completed",
            text=text,
            output=output_messages,
        )

        # Emit response.completed
        yield OpenAIResponseObjectStreamResponseCompleted(response=final_response)

        if store:
            await self._store_response(
                response=final_response,
                input=input,
            )

    async def delete_openai_response(self, response_id: str) -> OpenAIDeleteResponseObject:
        return await self.responses_store.delete_response_object(response_id)

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
        from llama_stack.apis.tools import Tool

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
            elif input_tool.type in WebSearchToolTypes:
                tool_name = "web_search"
                tool = await self.tool_groups_api.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")
                chat_tools.append(make_openai_tool(tool_name, tool))
            elif input_tool.type == "file_search":
                tool_name = "knowledge_search"
                tool = await self.tool_groups_api.get_tool(tool_name)
                if not tool:
                    raise ValueError(f"Tool {tool_name} not found")
                chat_tools.append(make_openai_tool(tool_name, tool))
            elif input_tool.type == "mcp":
                from llama_stack.providers.utils.tools.mcp import list_mcp_tools

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

    async def _execute_knowledge_search_via_vector_store(
        self,
        query: str,
        response_file_search_tool: OpenAIResponseInputToolFileSearch,
    ) -> ToolInvocationResult:
        """Execute knowledge search using vector_stores.search API with filters support."""
        search_results = []

        # Create search tasks for all vector stores
        async def search_single_store(vector_store_id):
            try:
                search_response = await self.vector_io_api.openai_search_vector_store(
                    vector_store_id=vector_store_id,
                    query=query,
                    filters=response_file_search_tool.filters,
                    max_num_results=response_file_search_tool.max_num_results,
                    ranking_options=response_file_search_tool.ranking_options,
                    rewrite_query=False,
                )
                return search_response.data
            except Exception as e:
                logger.warning(f"Failed to search vector store {vector_store_id}: {e}")
                return []

        # Run all searches in parallel using gather
        search_tasks = [search_single_store(vid) for vid in response_file_search_tool.vector_store_ids]
        all_results = await asyncio.gather(*search_tasks)

        # Flatten results
        for results in all_results:
            search_results.extend(results)

        # Convert search results to tool result format matching memory.py
        # Format the results as interleaved content similar to memory.py
        content_items = []
        content_items.append(
            TextContentItem(
                text=f"knowledge_search tool found {len(search_results)} chunks:\nBEGIN of knowledge_search tool results.\n"
            )
        )

        for i, result_item in enumerate(search_results):
            chunk_text = result_item.content[0].text if result_item.content else ""
            metadata_text = f"document_id: {result_item.file_id}, score: {result_item.score}"
            if result_item.attributes:
                metadata_text += f", attributes: {result_item.attributes}"
            text_content = f"[{i + 1}] {metadata_text}\n{chunk_text}\n"
            content_items.append(TextContentItem(text=text_content))

        content_items.append(TextContentItem(text="END of knowledge_search tool results.\n"))
        content_items.append(
            TextContentItem(
                text=f'The above results were retrieved to help answer the user\'s query: "{query}". Use them as supporting information only in answering this query.\n',
            )
        )

        return ToolInvocationResult(
            content=content_items,
            metadata={
                "document_ids": [r.file_id for r in search_results],
                "chunks": [r.content[0].text if r.content else "" for r in search_results],
                "scores": [r.score for r in search_results],
            },
        )

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
        tool_kwargs = json.loads(function.arguments) if function.arguments else {}

        if not function or not tool_call_id or not function.name:
            return None, None

        error_exc = None
        result = None
        try:
            if ctx.mcp_tool_to_server and function.name in ctx.mcp_tool_to_server:
                from llama_stack.providers.utils.tools.mcp import invoke_mcp_tool

                mcp_tool = ctx.mcp_tool_to_server[function.name]
                result = await invoke_mcp_tool(
                    endpoint=mcp_tool.server_url,
                    headers=mcp_tool.headers or {},
                    tool_name=function.name,
                    kwargs=tool_kwargs,
                )
            elif function.name == "knowledge_search":
                response_file_search_tool = next(
                    (t for t in ctx.response_tools if isinstance(t, OpenAIResponseInputToolFileSearch)), None
                )
                if response_file_search_tool:
                    # Use vector_stores.search API instead of knowledge_search tool
                    # to support filters and ranking_options
                    query = tool_kwargs.get("query", "")
                    result = await self._execute_knowledge_search_via_vector_store(
                        query=query,
                        response_file_search_tool=response_file_search_tool,
                    )
            else:
                result = await self.tool_runtime_api.invoke_tool(
                    tool_name=function.name,
                    kwargs=tool_kwargs,
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
            elif function.name == "knowledge_search":
                message = OpenAIResponseOutputMessageFileSearchToolCall(
                    id=tool_call_id,
                    queries=[tool_kwargs.get("query", "")],
                    status="completed",
                )
                if "document_ids" in result.metadata:
                    message.results = []
                    for i, doc_id in enumerate(result.metadata["document_ids"]):
                        text = result.metadata["chunks"][i] if "chunks" in result.metadata else None
                        score = result.metadata["scores"][i] if "scores" in result.metadata else None
                        message.results.append(
                            {
                                "file_id": doc_id,
                                "filename": doc_id,
                                "text": text,
                                "score": score,
                            }
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
        else:
            text = str(error_exc)
            input_message = OpenAIToolMessageParam(content=text, tool_call_id=tool_call_id)

        return message, input_message


def _is_function_tool_call(
    tool_call: OpenAIChatCompletionToolCall,
    tools: list[OpenAIResponseInputTool],
) -> bool:
    if not tool_call.function:
        return False
    for t in tools:
        if t.type == "function" and t.name == tool_call.function.name:
            return True
    return False
