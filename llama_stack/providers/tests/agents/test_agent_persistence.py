# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
import pytest_asyncio

from llama_stack.apis.agents import *  # noqa: F403
from llama_stack.providers.tests.resolver import resolve_impls_for_test
from llama_stack.providers.datatypes import *  # noqa: F403

from dotenv import load_dotenv

from llama_stack.providers.utils.kvstore import kvstore_impl, SqliteKVStoreConfig

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
#   pytest -s llama_stack/providers/tests/agents/test_agent_persistence.py \
#   --tb=short --disable-warnings
# ```

load_dotenv()


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
async def test_delete_agents_and_sessions(agents_settings, sample_messages):
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
    persistence_store = await kvstore_impl(agents_settings["persistence"])

    await agents_impl.delete_agents_session(agent_id, session_id)
    session_response = await persistence_store.get(f"session:{agent_id}:{session_id}")

    await agents_impl.delete_agents(agent_id)
    agent_response = await persistence_store.get(f"agent:{agent_id}")

    assert session_response is None
    assert agent_response is None


async def test_get_agent_turns_and_steps(agents_settings, sample_messages):
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
        chunk async for chunk in await agents_impl.create_agent_turn(**turn_request)
    ]

    final_event = turn_response[-1].event.payload
    turn_id = final_event.turn.turn_id
    persistence_store = await kvstore_impl(SqliteKVStoreConfig())
    turn = await persistence_store.get(f"session:{agent_id}:{session_id}:{turn_id}")
    response = await agents_impl.get_agents_turn(agent_id, session_id, turn_id)

    assert isinstance(response, Turn)
    assert response == final_event.turn
    assert turn == final_event.turn

    steps = final_event.turn.steps
    step_id = steps[0].step_id
    step_response = await agents_impl.get_agents_step(
        agent_id, session_id, turn_id, step_id
    )

    assert isinstance(step_response.step, Step)
    assert step_response.step == steps[0]
