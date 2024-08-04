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

from llama_models.llama3_1.api.datatypes import BuiltinTool, SamplingParams

from .api import (
    AgenticSystem,
    AgenticSystemCreateRequest,
    AgenticSystemCreateResponse,
    AgenticSystemInstanceConfig,
    AgenticSystemSessionCreateRequest,
    AgenticSystemSessionCreateResponse,
    AgenticSystemToolDefinition,
    AgenticSystemTurnCreateRequest,
    AgenticSystemTurnResponseStreamChunk,
)


async def get_client_impl(base_url: str):
    return AgenticSystemClient(base_url)


class AgenticSystemClient(AgenticSystem):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def create_agentic_system(
        self, request: AgenticSystemCreateRequest
    ) -> AgenticSystemCreateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agentic_system/create",
                data=request.json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return AgenticSystemCreateResponse(**response.json())

    async def create_agentic_system_session(
        self,
        request: AgenticSystemSessionCreateRequest,
    ) -> AgenticSystemSessionCreateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agentic_system/session/create",
                data=request.json(),
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
                data=request.json(),
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
    api = await AgenticSystemClient(f"http://{host}:{port}")

    tool_definitions = [
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.brave_search,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.wolfram_alpha,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.photogen,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.code_interpreter,
        ),
    ]

    create_request = AgenticSystemCreateRequest(
        model="Meta-Llama3.1-8B-Instruct",
        instance_config=AgenticSystemInstanceConfig(
            instructions="You are a helpful assistant",
            sampling_params=SamplingParams(),
            available_tools=tool_definitions,
            input_shields=[],
            output_shields=[],
            quantization_config=None,
            debug_prefix_messages=[],
        ),
    )

    create_response = await api.create_agentic_system(create_request)
    print(create_response)
    # TODO: Add chat session / turn apis to test e2e


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
