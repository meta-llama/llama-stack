# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import AsyncIterator, List, Optional, Union
from unittest.mock import MagicMock

import pytest

from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from llama_stack.apis.agents import *  # noqa: F403

from ..agent_instance import ChatAgent


class MockInferenceAPI:
    async def chat_completion(
        self,
        model: str,
        messages: List[Message],
        sampling_params: Optional[SamplingParams] = SamplingParams(),
        tools: Optional[List[ToolDefinition]] = None,
        tool_choice: Optional[ToolChoice] = None,
        tool_prompt_format: Optional[ToolPromptFormat] = None,
        stream: Optional[bool] = False,
        logprobs: Optional[LogProbConfig] = None,
    ) -> AsyncIterator[
        Union[ChatCompletionResponseStreamChunk, ChatCompletionResponse]
    ]:
        if stream:
            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type="start",
                    delta="",
                )
            )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type="progress",
                    delta="Mock response",
                )
            )

            yield ChatCompletionResponseStreamChunk(
                event=ChatCompletionResponseEvent(
                    event_type="complete",
                    delta="",
                    stop_reason="end_of_turn",
                )
            )
        else:
            yield ChatCompletionResponse(
                completion_message=CompletionMessage(
                    role="assistant", content="Mock response", stop_reason="end_of_turn"
                ),
                logprobs=[0.1, 0.2, 0.3] if logprobs else None,
            )


class MockSafetyAPI:
    async def run_shields(
        self, messages: List[Message], shields: List[MagicMock]
    ) -> List[ShieldResponse]:
        return [ShieldResponse(shield_type="mock_shield", is_violation=False)]


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
def chat_agent(mock_inference_api, mock_safety_api, mock_memory_api):
    # You'll need to adjust this based on the actual ChatAgent constructor
    agent_config = AgentConfig(
        model="test_model",
        instructions="You are a helpful assistant.",
        sampling_params=SamplingParams(),
        tools=[],
        tool_choice=ToolChoice.auto,
        input_shields=[],
        output_shields=[],
    )
    return ChatAgent(
        agent_config=agent_config,
        inference_api=mock_inference_api,
        memory_api=mock_memory_api,
        safety_api=mock_safety_api,
        builtin_tools=[],
    )


@pytest.mark.asyncio
async def test_chat_agent_create_session(chat_agent):
    session = chat_agent.create_session("Test Session")
    assert session.session_name == "Test Session"
    assert session.turns == []
    assert session.session_id in chat_agent.sessions


@pytest.mark.asyncio
async def test_chat_agent_create_and_execute_turn(chat_agent):
    session = chat_agent.create_session("Test Session")
    request = AgentTurnCreateRequest(
        agent_id="random",
        session_id=session.session_id,
        messages=[UserMessage(content="Hello")],
    )

    responses = []
    async for response in chat_agent.create_and_execute_turn(request):
        responses.append(response)

    print(responses)
    assert len(responses) > 0
    assert len(responses) == 4  # TurnStart, StepStart, StepComplete, TurnComplete
    assert responses[0].event.payload.turn_id is not None


@pytest.mark.asyncio
async def test_run_shields_wrapper(chat_agent):
    messages = [UserMessage(content="Test message")]
    shields = [ShieldDefinition(shield_type="test_shield")]

    responses = [
        chunk
        async for chunk in chat_agent.run_shields_wrapper(
            turn_id="test_turn_id",
            messages=messages,
            shields=shields,
            touchpoint="user-input",
        )
    ]

    assert len(responses) == 2  # StepStart, StepComplete
    assert responses[0].event.payload.step_type.value == "shield_call"
    assert not responses[1].event.payload.step_details.response.is_violation
