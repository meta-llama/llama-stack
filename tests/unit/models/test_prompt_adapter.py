# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import pytest

from llama_stack.apis.inference import (
    ChatCompletionRequest,
    CompletionMessage,
    StopReason,
    SystemMessage,
    ToolCall,
    ToolConfig,
    UserMessage,
)
from llama_stack.models.llama.datatypes import (
    CodeInterpreterTool,
    FunctionTool,
    ToolParamDefinition,
    ToolPromptFormat,
    WebSearchTool,
)
from llama_stack.providers.utils.inference.prompt_adapter import (
    chat_completion_request_to_messages,
    chat_completion_request_to_prompt,
)

MODEL = "Llama3.1-8B-Instruct"
MODEL3_2 = "Llama3.2-3B-Instruct"


@pytest.fixture(autouse=True)
def setup_loop():
    loop = asyncio.get_event_loop()
    loop.set_debug(False)
    return loop


@pytest.mark.asyncio
async def test_system_default():
    content = "Hello !"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            UserMessage(content=content),
        ],
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 2
    assert messages[-1].content == content
    assert "Cutting Knowledge Date: December 2023" in messages[0].content


@pytest.mark.asyncio
async def test_system_builtin_only():
    content = "Hello !"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            UserMessage(content=content),
        ],
        tools=[
            CodeInterpreterTool(),
            WebSearchTool(),
        ],
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 2
    assert messages[-1].content == content
    assert "Cutting Knowledge Date: December 2023" in messages[0].content
    assert "Tools: brave_search" in messages[0].content


@pytest.mark.asyncio
async def test_system_custom_only():
    content = "Hello !"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            UserMessage(content=content),
        ],
        tools=[
            FunctionTool(
                name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            )
        ],
        tool_config=ToolConfig(tool_prompt_format=ToolPromptFormat.json),
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 3
    assert "Environment: ipython" in messages[0].content
    assert "Return function calls in JSON format" in messages[1].content
    assert messages[-1].content == content


@pytest.mark.asyncio
async def test_system_custom_and_builtin():
    content = "Hello !"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            UserMessage(content=content),
        ],
        tools=[
            CodeInterpreterTool(),
            WebSearchTool(),
            FunctionTool(
                name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            ),
        ],
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 3
    assert "Environment: ipython" in messages[0].content
    assert "Tools: brave_search" in messages[0].content
    assert "Return function calls in JSON format" in messages[1].content
    assert messages[-1].content == content


@pytest.mark.asyncio
async def test_completion_message_encoding():
    request = ChatCompletionRequest(
        model=MODEL3_2,
        messages=[
            UserMessage(content="hello"),
            CompletionMessage(
                content="",
                stop_reason=StopReason.end_of_turn,
                tool_calls=[
                    ToolCall(
                        type="function",
                        tool_name="custom1",
                        arguments={"param1": "value1"},
                        call_id="123",
                    )
                ],
            ),
        ],
        tools=[
            FunctionTool(
                name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            ),
        ],
        tool_config=ToolConfig(tool_prompt_format=ToolPromptFormat.python_list),
    )
    prompt = await chat_completion_request_to_prompt(request, request.model)
    assert '[custom1(param1="value1")]' in prompt

    request.model = MODEL
    request.tool_config.tool_prompt_format = ToolPromptFormat.json
    prompt = await chat_completion_request_to_prompt(request, request.model)
    assert '{"type": "function", "name": "custom1", "parameters": {"param1": "value1"}}' in prompt


@pytest.mark.asyncio
async def test_user_provided_system_message():
    content = "Hello !"
    system_prompt = "You are a pirate"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=content),
        ],
        tools=[
            CodeInterpreterTool(),
        ],
    )
    messages = chat_completion_request_to_messages(request, MODEL)
    assert len(messages) == 2
    assert messages[0].content.endswith(system_prompt)
    assert messages[-1].content == content


@pytest.mark.asyncio
async def test_repalce_system_message_behavior_builtin_tools():
    content = "Hello !"
    system_prompt = "You are a pirate"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=content),
        ],
        tools=[
            CodeInterpreterTool(),
        ],
        tool_config=ToolConfig(
            tool_choice="auto",
            tool_prompt_format="python_list",
            system_message_behavior="replace",
        ),
    )
    messages = chat_completion_request_to_messages(request, MODEL3_2)
    assert len(messages) == 2
    assert messages[0].content.endswith(system_prompt)
    assert "Environment: ipython" in messages[0].content
    assert messages[-1].content == content


@pytest.mark.asyncio
async def test_repalce_system_message_behavior_custom_tools():
    content = "Hello !"
    system_prompt = "You are a pirate"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=content),
        ],
        tools=[
            CodeInterpreterTool(),
            FunctionTool(
                name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            ),
        ],
        tool_config=ToolConfig(
            tool_choice="auto",
            tool_prompt_format="python_list",
            system_message_behavior="replace",
        ),
    )
    messages = chat_completion_request_to_messages(request, MODEL3_2)
    assert len(messages) == 2
    assert messages[0].content.endswith(system_prompt)
    assert "Environment: ipython" in messages[0].content
    assert messages[-1].content == content


@pytest.mark.asyncio
async def test_replace_system_message_behavior_custom_tools_with_template():
    content = "Hello !"
    system_prompt = "You are a pirate {{ function_description }}"
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=content),
        ],
        tools=[
            CodeInterpreterTool(),
            FunctionTool(
                name="custom1",
                description="custom1 tool",
                parameters={
                    "param1": ToolParamDefinition(
                        param_type="str",
                        description="param1 description",
                        required=True,
                    ),
                },
            ),
        ],
        tool_config=ToolConfig(
            tool_choice="auto",
            tool_prompt_format="python_list",
            system_message_behavior="replace",
        ),
    )
    messages = chat_completion_request_to_messages(request, MODEL3_2)
    assert len(messages) == 2
    assert "Environment: ipython" in messages[0].content
    assert "You are a pirate" in messages[0].content
    assert '"name": "custom1"' in messages[0].content
    assert messages[-1].content == content
