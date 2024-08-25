# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from typing import AsyncGenerator

import fire

import httpx

from pydantic import BaseModel
from termcolor import cprint

from llama_models.llama3.api.datatypes import *  # noqa: F403
from .api import *  # noqa: F403

from .event_logger import EventLogger


async def get_client_impl(base_url: str):
    return AgenticSystemClient(base_url)


def encodable_dict(d: BaseModel):
    return json.loads(d.json())


class AgenticSystemClient(AgenticSystem):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def create_agentic_system(
        self, agent_config: AgentConfig
    ) -> AgenticSystemCreateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agentic_system/create",
                json={
                    "agent_config": encodable_dict(agent_config),
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return AgenticSystemCreateResponse(**response.json())

    async def create_agentic_system_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgenticSystemSessionCreateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agentic_system/session/create",
                json={
                    "agent_id": agent_id,
                    "session_name": session_name,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return AgenticSystemSessionCreateResponse(**response.json())

    async def create_agentic_system_turn(
        self,
        request: AgenticSystemTurnCreateRequest,
    ) -> AsyncGenerator:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/agentic_system/turn/create",
                json={
                    "request": encodable_dict(request),
                },
                headers={"Content-Type": "application/json"},
                timeout=20,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data: ") :]
                        try:
                            yield AgenticSystemTurnResponseStreamChunk(
                                **json.loads(data)
                            )
                        except Exception as e:
                            print(data)
                            print(f"Error with parsing or validation: {e}")


async def run_main(host: str, port: int):
    # client to test remote impl of agentic system
    api = AgenticSystemClient(f"http://{host}:{port}")

    tool_definitions = [
        BraveSearchToolDefinition(),
        WolframAlphaToolDefinition(),
        CodeInterpreterToolDefinition(),
    ]
    tool_definitions += [
        FunctionCallToolDefinition(
            function_name="get_boiling_point",
            description="Get the boiling point of a imaginary liquids (eg. polyjuice)",
            parameters={
                "liquid_name": ToolParamDefinition(
                    param_type="str",
                    description="The name of the liquid",
                    required=True,
                ),
                "celcius": ToolParamDefinition(
                    param_type="str",
                    description="Whether to return the boiling point in Celcius",
                    required=False,
                ),
            },
        ),
    ]

    agent_config = AgentConfig(
        model="Meta-Llama3.1-8B-Instruct",
        instructions="You are a helpful assistant",
        sampling_params=SamplingParams(temperature=1.0, top_p=0.9),
        tools=tool_definitions,
        tool_choice=ToolChoice.auto,
        tool_prompt_format=ToolPromptFormat.function_tag,
    )

    create_response = await api.create_agentic_system(agent_config)
    print(create_response)

    session_response = await api.create_agentic_system_session(
        agent_id=create_response.agent_id,
        session_name="test_session",
    )
    print(session_response)

    user_prompts = [
        "Who are you?",
        "what is the 100th prime number?",
        "Search web for who was 44th President of USA?",
        "Write code to check if a number is prime. Use that to check if 7 is prime",
        "What is the boiling point of polyjuicepotion ?",
    ]
    for content in user_prompts:
        cprint(f"User> {content}", color="blue")
        iterator = api.create_agentic_system_turn(
            AgenticSystemTurnCreateRequest(
                agent_id=create_response.agent_id,
                session_id=session_response.session_id,
                messages=[
                    UserMessage(content=content),
                ],
                stream=True,
            )
        )

        async for event, log in EventLogger().log(iterator):
            if log is not None:
                log.print()


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
