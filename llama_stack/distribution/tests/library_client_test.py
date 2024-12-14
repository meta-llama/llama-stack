# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger as AgentEventLogger
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import Attachment, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig


def main(config_path: str):
    client = LlamaStackAsLibraryClient(config_path)
    if not client.initialize():
        return

    models = client.models.list()
    print("\nModels:")
    for model in models:
        print(model)

    if not models:
        print("No models found, skipping chat completion test")
        return

    model_id = models[0].identifier
    response = client.inference.chat_completion(
        messages=[UserMessage(content="What is the capital of France?", role="user")],
        model_id=model_id,
        stream=False,
    )
    print("\nChat completion response (non-stream):")
    print(response)

    response = client.inference.chat_completion(
        messages=[UserMessage(content="What is the capital of France?", role="user")],
        model_id=model_id,
        stream=True,
    )

    print("\nChat completion response (stream):")
    for log in EventLogger().log(response):
        log.print()

    print("\nAgent test:")
    agent_config = AgentConfig(
        model=model_id,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=(
            [
                {
                    "type": "brave_search",
                    "engine": "brave",
                    "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
                }
            ]
            if os.getenv("BRAVE_SEARCH_API_KEY")
            else []
        )
        + (
            [
                {
                    "type": "code_interpreter",
                }
            ]
        ),
        tool_choice="required",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )
    agent = Agent(client, agent_config)
    user_prompts = [
        "Hello",
        "Which players played in the winning team of the NBA western conference semifinals of 2024, please use tools",
    ]
    user_prompts = [
        (
            "Here is a csv, can you describe it ?",
            [
                Attachment(
                    content="https://raw.githubusercontent.com/meta-llama/llama-stack-apps/main/examples/resources/inflation.csv",
                    mime_type="test/csv",
                )
            ],
        ),
        ("Which year ended with the highest inflation ?", None),
        (
            "What macro economic situations that led to such high inflation in that period?",
            None,
        ),
        ("Plot average yearly inflation as a time series", None),
    ]

    session_id = agent.create_session("test-session")

    for prompt, attachments in user_prompts:
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            attachments=attachments,
            session_id=session_id,
        )

        for log in AgentEventLogger().log(response):
            log.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()
    main(args.config_path)
