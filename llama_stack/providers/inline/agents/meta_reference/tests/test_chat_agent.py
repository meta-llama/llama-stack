# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
from typing import AsyncIterator, List, Optional, Union

import pytest

from llama_stack.apis.agents import (
    AgentConfig,
    AgentToolGroupWithArgs,
    AgentTurnCreateRequest,
    AgentTurnResponseTurnCompletePayload,
    StepType,
)
from llama_stack.apis.common.content_types import URL
from llama_stack.apis.inference import (
    ChatCompletionResponse,
    ChatCompletionResponseEvent,
    ChatCompletionResponseStreamChunk,
    CompletionMessage,
    LogProbConfig,
    Message,
    ResponseFormat,
    SamplingParams,
    ToolChoice,
    ToolDefinition,
    ToolPromptFormat,
    UserMessage,
)
from llama_stack.apis.safety import RunShieldResponse
from llama_stack.apis.tools import (
    Tool,
    ToolDef,
    ToolGroup,
    ToolHost,
    ToolInvocationResult,
)
from llama_stack.apis.vector_io import QueryChunksResponse
from llama_stack.models.llama.datatypes import BuiltinTool
from llama_stack.providers.inline.agents.meta_reference.agent_instance import (
    MEMORY_QUERY_TOOL,
)
from llama_stack.providers.inline.agents.meta_reference.agents import (
    MetaReferenceAgentsImpl,
    MetaReferenceAgentsImplConfig,
)
from llama_stack.providers.utils.kvstore.config import SqliteKVStoreConfig


class MockInferenceAPI:
    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        response_format: Optional[ResponseFormat] = None,
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> Union[ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]]:
        async def stream_response():
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type="start",
                    delta="",
                )
            )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type="progress",
                    delta="AI is a fascinating field...",
                )
            )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type="complete",
                    delta="",
                    stop_reason="end_of_turn",
                )
            )

        if stream:
            return stream_response()
        else:
            return ChatCompletionResponse(
                completion_message=CompletionMessage(
                    role="assistant",
                    content="Mock response",
                    stop_reason="end_of_turn",
                ),
                logprobs={"token_logprobs": [0.1, 0.2, 0.3]} if logprobs else None,
            )


class MockSafetyAPI:
    async def run_shield(self, shield_id: str, messages: List[Message]) -> RunShieldResponse:
        return RunShieldResponse(violation=None)


class MockVectorIOAPI:
    def __init__(self):
        self.chunks = {}

    async def insert_chunks(self, vector_db_id, chunks, ttl_seconds=None):
        for chunk in chunks:
            metadata = chunk.metadata
            self.chunks[vector_db_id][metadata["document_id"]] = chunk

    async def query_chunks(self, vector_db_id, query, params=None):
        if vector_db_id not in self.chunks:
            raise ValueError(f"Bank {vector_db_id} not found")

        chunks = list(self.chunks[vector_db_id].values())
        scores = [1.0] * len(chunks)
        return QueryChunksResponse(chunks=chunks, scores=scores)


class MockToolGroupsAPI:
    async def register_tool_group(self, toolgroup_id: str, provider_id: str, mcp_endpoint=None, args=None) -> None:
        pass

    async def get_tool_group(self, toolgroup_id: str) -> ToolGroup:
        return ToolGroup(
            identifier=toolgroup_id,
            provider_resource_id=toolgroup_id,
        )

    async def list_tool_groups(self) -> List[ToolGroup]:
        return []

    async def list_tools(self, tool_group_id: Optional[str] = None) -> List[Tool]:
        if tool_group_id == MEMORY_TOOLGROUP:
            return [
                Tool(
                    identifier=MEMORY_QUERY_TOOL,
                    provider_resource_id=MEMORY_QUERY_TOOL,
                    toolgroup_id=MEMORY_TOOLGROUP,
                    tool_host=ToolHost.client,
                    description="Mock tool",
                    provider_id="builtin::rag",
                    parameters=[],
                )
            ]
        if tool_group_id == CODE_INTERPRETER_TOOLGROUP:
            return [
                Tool(
                    identifier="code_interpreter",
                    provider_resource_id="code_interpreter",
                    toolgroup_id=CODE_INTERPRETER_TOOLGROUP,
                    tool_host=ToolHost.client,
                    description="Mock tool",
                    provider_id="builtin::code_interpreter",
                    parameters=[],
                )
            ]
        return []

    async def get_tool(self, tool_name: str) -> Tool:
        return Tool(
            identifier=tool_name,
            provider_resource_id=tool_name,
            toolgroup_id="mock_group",
            tool_host=ToolHost.client,
            description="Mock tool",
            provider_id="mock_provider",
            parameters=[],
        )

    async def unregister_tool_group(self, tool_group_id: str) -> None:
        pass


class MockToolRuntimeAPI:
    async def list_runtime_tools(
        self, tool_group_id: Optional[str] = None, mcp_endpoint: Optional[URL] = None
    ) -> List[ToolDef]:
        return []

    async def invoke_tool(self, tool_name: str, args: dict) -> ToolInvocationResult:
        return ToolInvocationResult(content={"result": "Mock tool result"})


@pytest.fixture
def mock_inference_api():
    return MockInferenceAPI()


@pytest.fixture
def mock_safety_api():
    return MockSafetyAPI()


@pytest.fixture
def mock_vector_io_api():
    return MockVectorIOAPI()


@pytest.fixture
def mock_tool_groups_api():
    return MockToolGroupsAPI()


@pytest.fixture
def mock_tool_runtime_api():
    return MockToolRuntimeAPI()


@pytest.fixture
async def get_agents_impl(
    mock_inference_api,
    mock_safety_api,
    mock_vector_io_api,
    mock_tool_runtime_api,
    mock_tool_groups_api,
):
    sqlite_file = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    impl = MetaReferenceAgentsImpl(
        config=MetaReferenceAgentsImplConfig(
            persistence_store=SqliteKVStoreConfig(
                db_name=sqlite_file.name,
            ),
        ),
        inference_api=mock_inference_api,
        safety_api=mock_safety_api,
        vector_io_api=mock_vector_io_api,
        tool_runtime_api=mock_tool_runtime_api,
        tool_groups_api=mock_tool_groups_api,
    )
    await impl.initialize()
    return impl


@pytest.fixture
async def get_chat_agent(get_agents_impl):
    impl = await get_agents_impl
    agent_config = AgentConfig(
        model="test_model",
        instructions="You are a helpful assistant.",
        toolgroups=[],
        tool_choice=ToolChoice.auto,
        enable_session_persistence=False,
        input_shields=["test_shield"],
    )
    response = await impl.create_agent(agent_config)
    return await impl.get_agent(response.agent_id)


MEMORY_TOOLGROUP = "builtin::rag"
CODE_INTERPRETER_TOOLGROUP = "builtin::code_interpreter"


@pytest.fixture
async def get_chat_agent_with_tools(get_agents_impl, request):
    impl = await get_agents_impl
    toolgroups = request.param
    agent_config = AgentConfig(
        model="test_model",
        instructions="You are a helpful assistant.",
        toolgroups=toolgroups,
        tool_choice=ToolChoice.auto,
        enable_session_persistence=False,
        input_shields=["test_shield"],
    )
    response = await impl.create_agent(agent_config)
    return await impl.get_agent(response.agent_id)


@pytest.mark.asyncio
async def test_chat_agent_create_and_execute_turn(get_chat_agent):
    chat_agent = await get_chat_agent
    session_id = await chat_agent.create_session("Test Session")
    request = AgentTurnCreateRequest(
        agent_id=chat_agent.agent_id,
        session_id=session_id,
        messages=[UserMessage(content="Hello")],
        stream=True,
    )

    responses = []
    async for response in chat_agent.create_and_execute_turn(request):
        responses.append(response)

    assert len(responses) > 0
    assert (
        len(responses) == 7
    )  # TurnStart, ShieldCallStart, ShieldCallComplete, StepStart, StepProgress, StepComplete, TurnComplete
    assert responses[0].event.payload.turn_id is not None


@pytest.mark.asyncio
async def test_run_multiple_shields_wrapper(get_chat_agent):
    chat_agent = await get_chat_agent
    messages = [UserMessage(content="Test message")]
    shields = ["test_shield"]

    responses = [
        chunk
        async for chunk in chat_agent.run_multiple_shields_wrapper(
            turn_id="test_turn_id",
            messages=messages,
            shields=shields,
            touchpoint="user-input",
        )
    ]

    assert len(responses) == 2  # StepStart, StepComplete
    assert responses[0].event.payload.step_type.value == "shield_call"
    assert not responses[1].event.payload.step_details.violation


@pytest.mark.asyncio
async def test_chat_agent_complex_turn(get_chat_agent):
    chat_agent = await get_chat_agent
    session_id = await chat_agent.create_session("Test Session")
    request = AgentTurnCreateRequest(
        agent_id=chat_agent.agent_id,
        session_id=session_id,
        messages=[UserMessage(content="Tell me about AI and then use a tool.")],
        stream=True,
    )

    responses = []
    async for response in chat_agent.create_and_execute_turn(request):
        responses.append(response)

    assert len(responses) > 0

    step_types = [
        response.event.payload.step_type for response in responses if hasattr(response.event.payload, "step_type")
    ]

    assert StepType.shield_call in step_types, "Shield call step is missing"
    assert StepType.inference in step_types, "Inference step is missing"

    event_types = [
        response.event.payload.event_type for response in responses if hasattr(response.event.payload, "event_type")
    ]
    assert "turn_start" in event_types, "Start event is missing"
    assert "turn_complete" in event_types, "Complete event is missing"

    assert any(isinstance(response.event.payload, AgentTurnResponseTurnCompletePayload) for response in responses), (
        "Turn complete event is missing"
    )
    turn_complete_payload = next(
        response.event.payload
        for response in responses
        if isinstance(response.event.payload, AgentTurnResponseTurnCompletePayload)
    )
    turn = turn_complete_payload.turn
    assert turn.input_messages == request.messages, "Input messages do not match"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "toolgroups, expected_memory, expected_code_interpreter",
    [
        ([], False, False),  # no tools
        ([MEMORY_TOOLGROUP], True, False),  # memory only
        ([CODE_INTERPRETER_TOOLGROUP], False, True),  # code interpreter only
        ([MEMORY_TOOLGROUP, CODE_INTERPRETER_TOOLGROUP], True, True),  # all tools
    ],
)
async def test_chat_agent_tools(get_agents_impl, toolgroups, expected_memory, expected_code_interpreter):
    impl = await get_agents_impl
    agent_config = AgentConfig(
        model="test_model",
        instructions="You are a helpful assistant.",
        toolgroups=toolgroups,
        tool_choice=ToolChoice.auto,
        enable_session_persistence=False,
        input_shields=["test_shield"],
    )
    response = await impl.create_agent(agent_config)
    chat_agent = await impl.get_agent(response.agent_id)

    tool_defs, _ = await chat_agent._get_tool_defs()
    if expected_memory:
        assert MEMORY_QUERY_TOOL in tool_defs
    if expected_code_interpreter:
        assert BuiltinTool.code_interpreter in tool_defs
    if expected_memory and expected_code_interpreter:
        # override the tools for turn
        new_tool_defs, _ = await chat_agent._get_tool_defs(
            toolgroups_for_turn=[
                AgentToolGroupWithArgs(
                    name=MEMORY_TOOLGROUP,
                    args={"vector_dbs": ["test_vector_db"]},
                )
            ]
        )
        assert MEMORY_QUERY_TOOL in new_tool_defs
        assert BuiltinTool.code_interpreter not in new_tool_defs
