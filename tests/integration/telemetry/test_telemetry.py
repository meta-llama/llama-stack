# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import time
from uuid import uuid4

from llama_stack_client import Agent


def test_agent_query_spans(llama_stack_client, text_model_id):
    agent = Agent(llama_stack_client, model=text_model_id, instructions="You are a helpful assistant")
    session_id = agent.create_session(f"test-session-{uuid4()}")
    agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": "Give me a sentence that contains the word: hello",
            }
        ],
        session_id=session_id,
        stream=False,
    )

    # Wait for the span to be logged
    time.sleep(2)

    agent_logs = []

    for span in llama_stack_client.telemetry.query_spans(
        attribute_filters=[
            {"key": "session_id", "op": "eq", "value": session_id},
        ],
        attributes_to_return=["input", "output"],
    ):
        if span.attributes["output"] != "no shields":
            agent_logs.append(span.attributes)

    assert len(agent_logs) == 1
    assert "Give me a sentence that contains the word: hello" in agent_logs[0]["input"]
    assert "hello" in agent_logs[0]["output"].lower()


def test_agent_name_filtering(llama_stack_client, text_model_id):
    # Create an agent with a specific name
    agent_name = f"test-agent-{uuid4()}"
    print(f"Using agent_name: {agent_name}")

    agent = Agent(
        llama_stack_client,
        model=text_model_id,
        instructions="You are a helpful assistant",
        name=agent_name,
    )
    session_id = agent.create_session(f"test-session-{uuid4()}")
    print(f"Created session_id: {session_id}")

    agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": "Give me a sentence that contains the word: hello",
            }
        ],
        session_id=session_id,
        stream=False,
    )

    # Wait for the span to be logged - increase the time to ensure it's processed
    time.sleep(5)

    # Query spans filtered by session_id to see what's available
    all_spans = []
    for span in llama_stack_client.telemetry.query_spans(
        attribute_filters=[
            {"key": "session_id", "op": "eq", "value": session_id},
        ],
        attributes_to_return=["input", "output", "agent_name", "agent_id", "session_id"],
    ):
        all_spans.append(span.attributes)

    print(f"All spans for session {session_id}:")
    for span in all_spans:
        print(f"Span attributes: {span}")

    # Query all spans to see if any have the agent_name attribute
    agent_name_spans = []
    for span in llama_stack_client.telemetry.query_spans(
        attribute_filters=[],
        attributes_to_return=["agent_name"],
    ):
        if "agent_name" in span.attributes:
            agent_name_spans.append(span.attributes)

    print("All spans with agent_name attribute:")
    for span in agent_name_spans:
        print(f"Span with agent_name: {span}")

    # Query spans filtered by agent name
    agent_logs = []
    for span in llama_stack_client.telemetry.query_spans(
        attribute_filters=[
            {"key": "agent_name", "op": "eq", "value": agent_name},
        ],
        attributes_to_return=["input", "output", "agent_name"],
    ):
        if "output" in span.attributes and span.attributes["output"] != "no shields":
            agent_logs.append(span.attributes)

    print(f"Found {len(agent_logs)} spans filtered by agent_name")

    assert len(agent_logs) == 1
    assert agent_logs[0]["agent_name"] == agent_name
    assert "Give me a sentence that contains the word: hello" in agent_logs[0]["input"]
    assert "hello" in agent_logs[0]["output"].lower()
