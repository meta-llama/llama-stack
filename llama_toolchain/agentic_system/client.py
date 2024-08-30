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
from llama_toolchain.core.datatypes import RemoteProviderConfig

from .api import *  # noqa: F403
from .event_logger import EventLogger


async def get_client_impl(config: RemoteProviderConfig, _deps):
    return AgenticSystemClient(config.url)


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
                            if "error" in data:
                                cprint(data, "red")
                                continue

                            yield AgenticSystemTurnResponseStreamChunk(
                                **json.loads(data)
                            )
                        except Exception as e:
                            print(data)
                            print(f"Error with parsing or validation: {e}")


async def _run_agent(api, tool_definitions, user_prompts, attachments=None):
    agent_config = AgentConfig(
        model="Meta-Llama3.1-8B-Instruct",
        instructions="You are a helpful assistant",
        sampling_params=SamplingParams(temperature=1.0, top_p=0.9),
        tools=tool_definitions,
        tool_choice=ToolChoice.auto,
        tool_prompt_format=ToolPromptFormat.function_tag,
    )

    create_response = await api.create_agentic_system(agent_config)
    session_response = await api.create_agentic_system_session(
        agent_id=create_response.agent_id,
        session_name="test_session",
    )

    for content in user_prompts:
        cprint(f"User> {content}", color="white", attrs=["bold"])
        iterator = api.create_agentic_system_turn(
            AgenticSystemTurnCreateRequest(
                agent_id=create_response.agent_id,
                session_id=session_response.session_id,
                messages=[
                    UserMessage(content=content),
                ],
                attachments=attachments,
                stream=True,
            )
        )

        async for event, log in EventLogger().log(iterator):
            if log is not None:
                log.print()


async def run_main(host: str, port: int):
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

    user_prompts = [
        "Who are you?",
        "what is the 100th prime number?",
        "Search web for who was 44th President of USA?",
        "Write code to check if a number is prime. Use that to check if 7 is prime",
        "What is the boiling point of polyjuicepotion ?",
    ]
    await _run_agent(api, tool_definitions, user_prompts)


async def run_rag(host: str, port: int):
    api = AgenticSystemClient(f"http://{host}:{port}")

    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "datasets.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]
    attachments = [
        Attachment(
            content=URL(
                uri=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}"
            ),
            mime_type="text/plain",
        )
        for i, url in enumerate(urls)
    ]

    # Alternatively, you can pre-populate the memory bank with documents for example,
    # using `llama_toolchain.memory.client`. Then you can grab the bank_id
    # from the output of that run.
    tool_definitions = [
        MemoryToolDefinition(
            max_tokens_in_context=2048,
            memory_bank_configs=[],
        ),
    ]

    user_prompts = [
        "How do I use Lora?",
        "Tell me briefly about llama3 and torchtune",
    ]

    await _run_agent(api, tool_definitions, user_prompts, attachments)


def main(host: str, port: int, rag: bool = False):
    fn = run_rag if rag else run_main
    asyncio.run(fn(host, port))


if __name__ == "__main__":
    fire.Fire(main)
