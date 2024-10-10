# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.apis.agents import *  # noqa: F403
from llama_stack.providers.tests.resolver import resolve_impls_for_test

# How to run this test:
#
# 1. Ensure you have a conda environment with the right dependencies installed.
#    This includes `pytest` and `pytest-asyncio`.
#
# 2. Copy and modify the provider_config_example.yaml depending on the provider you are testing.
#
# 3. Run:
#
# ```bash
# PROVIDER_ID=<your_provider> \
#   PROVIDER_CONFIG=provider_config.yaml \
#   pytest -s llama_stack/providers/tests/agents/test_agents.py \
#   --tb=short --disable-warnings
# ```


@pytest_asyncio.fixture(scope="session")
async def agents_settings():
    impls = await resolve_impls_for_test(
        Api.agents, deps=[Api.inference, Api.memory, Api.safety]
    )

    return {
        "impl": impls[Api.agents],
        "memory_impl": impls[Api.memory],
        "common_params": {
            "model": "Llama3.1-8B-Instruct",
            "instructions": "You are a helpful assistant.",
        },
    }


@pytest.fixture
def sample_messages():
    return [
        UserMessage(content="What's the weather like today?"),
    ]


@pytest.mark.asyncio
async def test_create_agent_turn(agents_settings, sample_messages):
    agents_impl = agents_settings["impl"]

    # First, create an agent
    agent_config = AgentConfig(
        model=agents_settings["common_params"]["model"],
        instructions=agents_settings["common_params"]["instructions"],
        enable_session_persistence=True,
        sampling_params=SamplingParams(temperature=0.7, top_p=0.95),
        input_shields=[],
        output_shields=[],
        tools=[],
        max_infer_iters=5,
    )

    create_response = await agents_impl.create_agent(agent_config)
    agent_id = create_response.agent_id

    # Create a session
    session_create_response = await agents_impl.create_agent_session(
        agent_id, "Test Session"
    )
    session_id = session_create_response.session_id

    # Create and execute a turn
    turn_request = dict(
        agent_id=agent_id,
        session_id=session_id,
        messages=sample_messages,
        stream=True,
    )

    turn_response = [
        chunk async for chunk in agents_impl.create_agent_turn(**turn_request)
    ]

    assert len(turn_response) > 0
    assert all(
        isinstance(chunk, AgentTurnResponseStreamChunk) for chunk in turn_response
    )

    # Check for expected event types
    event_types = [chunk.event.payload.event_type for chunk in turn_response]
    assert AgentTurnResponseEventType.turn_start.value in event_types
    assert AgentTurnResponseEventType.step_start.value in event_types
    assert AgentTurnResponseEventType.step_complete.value in event_types
    assert AgentTurnResponseEventType.turn_complete.value in event_types

    # Check the final turn complete event
    final_event = turn_response[-1].event.payload
    assert isinstance(final_event, AgentTurnResponseTurnCompletePayload)
    assert isinstance(final_event.turn, Turn)
    assert final_event.turn.session_id == session_id
    assert final_event.turn.input_messages == sample_messages
    assert isinstance(final_event.turn.output_message, CompletionMessage)
    assert len(final_event.turn.output_message.content) > 0
