# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
import uuid
from collections.abc import AsyncIterator

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel

from llama_stack.apis.agents import Order
from llama_stack.apis.agents.openai_responses import (
    AllowedToolsFilter,
    ListOpenAIResponseInputItem,
    ListOpenAIResponseObject,
    MCPListToolsTool,
    OpenAIDeleteResponseObject,
    OpenAIResponseInput,
    OpenAIResponseInputMessageContentText,
    OpenAIResponseInputTool,
    OpenAIResponseInputToolMCP,
    OpenAIResponseMessage,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessageMCPListTools,
    OpenAIResponseText,
    OpenAIResponseTextFormat,
    WebSearchToolTypes,
)
from llama_stack.apis.inference import (
    Inference,
    OpenAISystemMessageParam,
)
from llama_stack.apis.tools import Tool, ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition
from llama_stack.providers.utils.inference.openai_compat import (
    convert_tooldef_to_openai_tool,
)
from llama_stack.providers.utils.responses.responses_store import ResponsesStore

from .streaming import StreamingResponseOrchestrator
from .tool_executor import ToolExecutor
from .types import ChatCompletionContext
from .utils import (
    convert_response_input_to_chat_messages,
    convert_response_text_to_chat_response_format,
)

logger = get_logger(name=__name__, category="responses")


class OpenAIResponsePreviousResponseWithInputItems(BaseModel):
    input_items: ListOpenAIResponseInputItem
    response: OpenAIResponseObject


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
        self.tool_executor = ToolExecutor(
            tool_groups_api=tool_groups_api,
            tool_runtime_api=tool_runtime_api,
            vector_io_api=vector_io_api,
        )

    async def _prepend_previous_response(
        self,
        input: str | list[OpenAIResponseInput],
        previous_response_id: str | None = None,
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
        include: list[str] | None = None,
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
        # Input preprocessing
        input = await self._prepend_previous_response(input, previous_response_id)
        messages = await convert_response_input_to_chat_messages(input)
        await self._prepend_instructions(messages, instructions)

        # Structured outputs
        response_format = await convert_response_text_to_chat_response_format(text)

        # Tool setup, TODO: refactor this slightly since this can also yield events
        chat_tools, mcp_tool_to_server, mcp_list_message = (
            await self._convert_response_tools_to_chat_tools(tools) if tools else (None, {}, None)
        )

        ctx = ChatCompletionContext(
            model=model,
            messages=messages,
            response_tools=tools,
            chat_tools=chat_tools,
            mcp_tool_to_server=mcp_tool_to_server,
            temperature=temperature,
            response_format=response_format,
        )

        # Create orchestrator and delegate streaming logic
        response_id = f"resp-{uuid.uuid4()}"
        created_at = int(time.time())

        orchestrator = StreamingResponseOrchestrator(
            inference_api=self.inference_api,
            ctx=ctx,
            response_id=response_id,
            created_at=created_at,
            text=text,
            max_infer_iters=max_infer_iters,
            tool_executor=self.tool_executor,
            mcp_list_message=mcp_list_message,
        )

        # Stream the response
        final_response = None
        async for stream_chunk in orchestrator.create_response():
            if stream_chunk.type == "response.completed":
                final_response = stream_chunk.response
            yield stream_chunk

        # Store the response if requested
        if store and final_response:
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
