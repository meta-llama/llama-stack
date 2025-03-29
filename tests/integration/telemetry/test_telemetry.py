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
