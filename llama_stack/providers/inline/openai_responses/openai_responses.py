# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import uuid
from typing import AsyncIterator, List, Optional, cast

from openai.types.chat import ChatCompletionToolParam

from llama_stack.apis.inference.inference import (
    Inference,
    OpenAIAssistantMessageParam,
    OpenAIChatCompletion,
    OpenAIChatCompletionContentPartTextParam,
    OpenAIChatCompletionToolCallFunction,
    OpenAIChoice,
    OpenAIMessageParam,
    OpenAIToolMessageParam,
    OpenAIUserMessageParam,
)
from llama_stack.apis.models.models import Models, ModelType
from llama_stack.apis.openai_responses import OpenAIResponses
from llama_stack.apis.openai_responses.openai_responses import (
    OpenAIResponseInputTool,
    OpenAIResponseObject,
    OpenAIResponseObjectStream,
    OpenAIResponseOutput,
    OpenAIResponseOutputMessage,
    OpenAIResponseOutputMessageContentOutputText,
    OpenAIResponseOutputMessageWebSearchToolCall,
)
from llama_stack.apis.tools.tools import ToolGroups, ToolInvocationResult, ToolRuntime
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import ToolDefinition, ToolParamDefinition
from llama_stack.providers.utils.inference.openai_compat import convert_tooldef_to_openai_tool
from llama_stack.providers.utils.kvstore import kvstore_impl

from .config import OpenAIResponsesImplConfig

logger = get_logger(name=__name__, category="openai_responses")

OPENAI_RESPONSES_PREFIX = "openai_responses:"


async def _previous_response_to_messages(previous_response: OpenAIResponseObject) -> List[OpenAIMessageParam]:
    messages: List[OpenAIMessageParam] = []
    for output_message in previous_response.output:
        if isinstance(output_message, OpenAIResponseOutputMessage):
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
    def __init__(
        self,
        config: OpenAIResponsesImplConfig,
        models_api: Models,
        inference_api: Inference,
        tool_groups_api: ToolGroups,
        tool_runtime_api: ToolRuntime,
    ):
        self.config = config
        self.models_api = models_api
        self.inference_api = inference_api
        self.tool_groups_api = tool_groups_api
        self.tool_runtime_api = tool_runtime_api

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
        tools: Optional[List[OpenAIResponseInputTool]] = None,
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

        chat_tools = await self._convert_response_tools_to_chat_tools(tools) if tools else None
        chat_response = await self.inference_api.openai_chat_completion(
            model=model_obj.identifier,
            messages=messages,
            tools=chat_tools,
        )
        # type cast to appease mypy
        chat_response = cast(OpenAIChatCompletion, chat_response)
        # dump and reload to map to our pydantic types
        chat_response = OpenAIChatCompletion.model_validate_json(chat_response.model_dump_json())

        output_messages: List[OpenAIResponseOutput] = []
        if chat_response.choices[0].finish_reason == "tool_calls":
            output_messages.extend(
                await self._execute_tool_and_return_final_output(model_obj.identifier, chat_response, messages)
            )
        else:
            output_messages.extend(await _openai_choices_to_output_messages(chat_response.choices))
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

    async def _convert_response_tools_to_chat_tools(
        self, tools: List[OpenAIResponseInputTool]
    ) -> List[ChatCompletionToolParam]:
        chat_tools: List[ChatCompletionToolParam] = []
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
        self, model_id: str, chat_response: OpenAIChatCompletion, messages: List[OpenAIMessageParam]
    ) -> List[OpenAIResponseOutput]:
        output_messages: List[OpenAIResponseOutput] = []
        choice = chat_response.choices[0]

        # If the choice is not an assistant message, we don't need to execute any tools
        if not isinstance(choice.message, OpenAIAssistantMessageParam):
            return output_messages

        # If the assistant message doesn't have any tool calls, we don't need to execute any tools
        if not choice.message.tool_calls:
            return output_messages

        # TODO: handle multiple tool calls
        function = choice.message.tool_calls[0].function

        # If the tool call is not a function, we don't need to execute it
        if not function:
            return output_messages

        # TODO: telemetry spans for tool calls
        result = await self._execute_tool_call(function)

        tool_call_prefix = "tc_"
        if function.name == "web_search":
            tool_call_prefix = "ws_"
        tool_call_id = f"{tool_call_prefix}{uuid.uuid4()}"

        # Handle tool call failure
        if not result:
            output_messages.append(
                OpenAIResponseOutputMessageWebSearchToolCall(
                    id=tool_call_id,
                    status="failed",
                )
            )
            return output_messages

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
    ) -> Optional[ToolInvocationResult]:
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
