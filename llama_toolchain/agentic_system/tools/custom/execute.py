# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, AsyncGenerator, List

from llama_models.llama3_1.api.datatypes import StopReason, ToolResponseMessage

from llama_toolchain.agentic_system.api import (
    AgenticSystem,
    AgenticSystemTurnCreateRequest,
    AgenticSystemTurnResponseEventType as EventType,
)

from llama_toolchain.inference.api import Message


async def execute_with_custom_tools(
    system: AgenticSystem,
    system_id: str,
    session_id: str,
    messages: List[Message],
    custom_tools: List[Any],
    max_iters: int = 5,
    stream: bool = True,
) -> AsyncGenerator:
    # first create a session, or do you keep a persistent session?
    tools_dict = {t.get_name(): t for t in custom_tools}

    current_messages = messages.copy()
    n_iter = 0
    while n_iter < max_iters:
        n_iter += 1

        request = AgenticSystemTurnCreateRequest(
            system_id=system_id,
            session_id=session_id,
            messages=current_messages,
            stream=stream,
        )

        turn = None
        async for chunk in system.create_agentic_system_turn(request):
            if chunk.event.payload.event_type != EventType.turn_complete.value:
                yield chunk
            else:
                turn = chunk.event.payload.turn

        message = turn.output_message
        if len(message.tool_calls) == 0:
            yield chunk
            return

        if message.stop_reason == StopReason.out_of_tokens:
            yield chunk
            return

        tool_call = message.tool_calls[0]
        if tool_call.tool_name not in tools_dict:
            m = ToolResponseMessage(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                content=f"Unknown tool `{tool_call.tool_name}` was called. Try again with something else",
            )
            next_message = m
        else:
            tool = tools_dict[tool_call.tool_name]
            result_messages = await execute_custom_tool(tool, message)
            next_message = result_messages[0]

        yield next_message
        current_messages = [next_message]


async def execute_custom_tool(tool: Any, message: Message) -> List[Message]:
    result_messages = await tool.run([message])
    assert (
        len(result_messages) == 1
    ), f"Expected single message, got {len(result_messages)}"

    return result_messages
