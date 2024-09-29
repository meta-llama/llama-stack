# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import os
from typing import AsyncGenerator

import fire
import httpx
from dotenv import load_dotenv

from pydantic import BaseModel
from termcolor import cprint

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.distribution.datatypes import RemoteProviderConfig

from .agents import *  # noqa: F403
from .event_logger import EventLogger


load_dotenv()


async def get_client_impl(config: RemoteProviderConfig, _deps):
    return AgentsClient(config.url)


def encodable_dict(d: BaseModel):
    return json.loads(d.json())


class AgentsClient(Agents):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def create_agent(self, agent_config: AgentConfig) -> AgentCreateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agents/create",
                json={
                    "agent_config": encodable_dict(agent_config),
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return AgentCreateResponse(**response.json())

    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agents/session/create",
                json={
                    "agent_id": agent_id,
                    "session_name": session_name,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return AgentSessionCreateResponse(**response.json())

    async def create_agent_turn(
        self,
        request: AgentTurnCreateRequest,
    ) -> AsyncGenerator:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/agents/turn/create",
                json=encodable_dict(request),
                headers={"Content-Type": "application/json"},
                timeout=20,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data: ") :]
                        try:
                            jdata = json.loads(data)
                            if "error" in jdata:
                                cprint(data, "red")
                                continue

                            yield AgentTurnResponseStreamChunk(**jdata)
                        except Exception as e:
                            print(data)
                            print(f"Error with parsing or validation: {e}")


async def _run_agent(
    api, model, tool_definitions, tool_prompt_format, user_prompts, attachments=None
):
    agent_config = AgentConfig(
        model=model,
        instructions="You are a helpful assistant",
        sampling_params=SamplingParams(temperature=0.6, top_p=0.9),
        tools=tool_definitions,
        tool_choice=ToolChoice.auto,
        tool_prompt_format=tool_prompt_format,
        enable_session_persistence=False,
    )

    create_response = await api.create_agent(agent_config)
    session_response = await api.create_agent_session(
        agent_id=create_response.agent_id,
        session_name="test_session",
    )

    for content in user_prompts:
        cprint(f"User> {content}", color="white", attrs=["bold"])
        iterator = api.create_agent_turn(
            AgentTurnCreateRequest(
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


async def run_llama_3_1(host: str, port: int):
    model = "Llama3.1-8B-Instruct"
    api = AgentsClient(f"http://{host}:{port}")

    tool_definitions = [
        SearchToolDefinition(
            engine=SearchEngineType.brave,
            api_key=os.getenv("BRAVE_SEARCH_API_KEY"),
        ),
        WolframAlphaToolDefinition(api_key=os.getenv("WOLFRAM_ALPHA_API_KEY")),
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
    await _run_agent(api, model, tool_definitions, ToolPromptFormat.json, user_prompts)


async def run_llama_3_2_rag(host: str, port: int):
    model = "Llama3.2-3B-Instruct"
    api = AgentsClient(f"http://{host}:{port}")

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
    # using `llama_stack.memory.client`. Then you can grab the bank_id
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

    await _run_agent(
        api, model, tool_definitions, ToolPromptFormat.json, user_prompts, attachments
    )


async def run_llama_3_2(host: str, port: int):
    model = "Llama3.2-3B-Instruct"
    api = AgentsClient(f"http://{host}:{port}")

    # zero shot tools for llama3.2 text models
    tool_definitions = [
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
                    param_type="bool",
                    description="Whether to return the boiling point in Celcius",
                    required=False,
                ),
            },
        ),
        FunctionCallToolDefinition(
            function_name="make_web_search",
            description="Search the web / internet for more realtime information",
            parameters={
                "query": ToolParamDefinition(
                    param_type="str",
                    description="the query to search for",
                    required=True,
                ),
            },
        ),
    ]

    user_prompts = [
        "Who are you?",
        "what is the 100th prime number?",
        "Who was 44th President of USA?",
        # multiple tool calls in a single prompt
        "What is the boiling point of polyjuicepotion and pinkponklyjuice?",
    ]
    await _run_agent(
        api, model, tool_definitions, ToolPromptFormat.python_list, user_prompts
    )


def main(host: str, port: int, run_type: str):
    assert run_type in [
        "tools_llama_3_1",
        "tools_llama_3_2",
        "rag_llama_3_2",
    ], f"Invalid run type {run_type}, must be one of tools_llama_3_1, tools_llama_3_2, rag_llama_3_2"

    fn = {
        "tools_llama_3_1": run_llama_3_1,
        "tools_llama_3_2": run_llama_3_2,
        "rag_llama_3_2": run_llama_3_2_rag,
    }
    asyncio.run(fn[run_type](host, port))


if __name__ == "__main__":
    fire.Fire(main)
