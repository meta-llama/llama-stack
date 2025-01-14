# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import tempfile
from typing import AsyncIterator, List, Optional, Union

import pytest
from llama_models.llama3.api.datatypes import BuiltinTool

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
    UserMessage,
)
from llama_stack.apis.memory import MemoryBank
from llama_stack.apis.memory_banks import BankParams, VectorMemoryBank
from llama_stack.apis.safety import RunShieldResponse
from llama_stack.apis.tools import (
    Tool,
    ToolDef,
    ToolGroup,
    ToolHost,
    ToolInvocationResult,
    ToolPromptFormat,
)
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
    ) -> Union[
        ChatCompletionResponse, AsyncIterator[ChatCompletionResponseStreamChunk]
    ]:
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
    async def run_shield(
        self, shield_id: str, messages: List[Message]
    ) -> RunShieldResponse:
        return RunShieldResponse(violation=None)


class MockMemoryAPI:
    def __init__(self):
        self.memory_banks = {}
        self.documents = {}

    async def create_memory_bank(self, name, config, url=None):
        bank_id = f"bank_{len(self.memory_banks)}"
        bank = MemoryBank(bank_id, name, config, url)
        self.memory_banks[bank_id] = bank
        self.documents[bank_id] = {}
        return bank

    async def list_memory_banks(self):
        return list(self.memory_banks.values())

    async def get_memory_bank(self, bank_id):
        return self.memory_banks.get(bank_id)

    async def drop_memory_bank(self, bank_id):
        if bank_id in self.memory_banks:
            del self.memory_banks[bank_id]
            del self.documents[bank_id]
        return bank_id

    async def insert_documents(self, bank_id, documents, ttl_seconds=None):
        if bank_id not in self.documents:
            raise ValueError(f"Bank {bank_id} not found")
        for doc in documents:
            self.documents[bank_id][doc.document_id] = doc

    async def update_documents(self, bank_id, documents):
        if bank_id not in self.documents:
            raise ValueError(f"Bank {bank_id} not found")
        for doc in documents:
            if doc.document_id in self.documents[bank_id]:
                self.documents[bank_id][doc.document_id] = doc

    async def query_documents(self, bank_id, query, params=None):
        if bank_id not in self.documents:
            raise ValueError(f"Bank {bank_id} not found")
        # Simple mock implementation: return all documents
        chunks = [
            {"content": doc.content, "token_count": 10, "document_id": doc.document_id}
            for doc in self.documents[bank_id].values()
        ]
        scores = [1.0] * len(chunks)
        return {"chunks": chunks, "scores": scores}

    async def get_documents(self, bank_id, document_ids):
        if bank_id not in self.documents:
            raise ValueError(f"Bank {bank_id} not found")
        return [
            self.documents[bank_id][doc_id]
            for doc_id in document_ids
            if doc_id in self.documents[bank_id]
        ]

    async def delete_documents(self, bank_id, document_ids):
        if bank_id not in self.documents:
            raise ValueError(f"Bank {bank_id} not found")
        for doc_id in document_ids:
            self.documents[bank_id].pop(doc_id, None)


class MockToolGroupsAPI:
    async def register_tool_group(
        self, toolgroup_id: str, provider_id: str, mcp_endpoint=None, args=None
    ) -> None:
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
                    provider_id="builtin::memory",
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


class MockMemoryBanksAPI:
    async def list_memory_banks(self) -> List[MemoryBank]:
        return []

    async def get_memory_bank(self, memory_bank_id: str) -> Optional[MemoryBank]:
        return None

    async def register_memory_bank(
        self,
        memory_bank_id: str,
        params: BankParams,
        provider_id: Optional[str] = None,
        provider_memory_bank_id: Optional[str] = None,
    ) -> MemoryBank:
        return VectorMemoryBank(
            identifier=memory_bank_id,
            provider_resource_id=provider_memory_bank_id or memory_bank_id,
            embedding_model="mock_model",
            chunk_size_in_tokens=512,
        )

    async def unregister_memory_bank(self, memory_bank_id: str) -> None:
        pass


@pytest.fixture
def mock_inference_api():
    return MockInferenceAPI()


@pytest.fixture
def mock_safety_api():
    return MockSafetyAPI()


@pytest.fixture
def mock_memory_api():
    return MockMemoryAPI()


@pytest.fixture
def mock_tool_groups_api():
    return MockToolGroupsAPI()


@pytest.fixture
def mock_tool_runtime_api():
    return MockToolRuntimeAPI()


@pytest.fixture
def mock_memory_banks_api():
    return MockMemoryBanksAPI()


@pytest.fixture
async def get_agents_impl(
    mock_inference_api,
    mock_safety_api,
    mock_memory_api,
    mock_memory_banks_api,
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
        memory_api=mock_memory_api,
        memory_banks_api=mock_memory_banks_api,
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


MEMORY_TOOLGROUP = "builtin::memory"
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
        response.event.payload.step_type
        for response in responses
        if hasattr(response.event.payload, "step_type")
    ]

    assert StepType.shield_call in step_types, "Shield call step is missing"
    assert StepType.inference in step_types, "Inference step is missing"

    event_types = [
        response.event.payload.event_type
        for response in responses
        if hasattr(response.event.payload, "event_type")
    ]
    assert "turn_start" in event_types, "Start event is missing"
    assert "turn_complete" in event_types, "Complete event is missing"

    assert any(
        isinstance(response.event.payload, AgentTurnResponseTurnCompletePayload)
        for response in responses
    ), "Turn complete event is missing"
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
async def test_chat_agent_tools(
    get_agents_impl, toolgroups, expected_memory, expected_code_interpreter
):
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
                    args={"memory_banks": ["test_memory_bank"]},
                )
            ]
        )
        assert MEMORY_QUERY_TOOL in new_tool_defs
        assert BuiltinTool.code_interpreter not in new_tool_defs
