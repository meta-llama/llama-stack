# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import Any, List, Optional

from llama_models.llama3_1.api.datatypes import BuiltinTool, Message, SamplingParams

from llama_toolchain.agentic_system.api import (
    AgenticSystemCreateRequest,
    AgenticSystemInstanceConfig,
    AgenticSystemSessionCreateRequest,
    AgenticSystemToolDefinition,
)
from llama_toolchain.agentic_system.client import AgenticSystemClient

from llama_toolchain.agentic_system.tools.custom.execute import (
    execute_with_custom_tools,
)
from llama_toolchain.safety.api.datatypes import BuiltinShield, ShieldDefinition


# TODO: this should move back to the llama-agentic-system repo


class AgenticSystemClientWrapper:

    def __init__(self, api, system_id, custom_tools):
        self.api = api
        self.system_id = system_id
        self.custom_tools = custom_tools
        self.session_id = None

    async def create_session(self, name: str = None):
        if name is None:
            name = f"Session-{uuid.uuid4()}"

        response = await self.api.create_agentic_system_session(
            AgenticSystemSessionCreateRequest(
                system_id=self.system_id,
                session_name=name,
            )
        )
        self.session_id = response.session_id
        return self.session_id

    async def run(self, messages: List[Message], stream: bool = True):
        async for chunk in execute_with_custom_tools(
            self.api,
            self.system_id,
            self.session_id,
            messages,
            self.custom_tools,
            stream=stream,
        ):
            yield chunk


async def get_agent_system_instance(
    host: str,
    port: int,
    custom_tools: Optional[List[Any]] = None,
    disable_safety: bool = False,
    model: str = "Meta-Llama3.1-8B-Instruct",
) -> AgenticSystemClientWrapper:
    custom_tools = custom_tools or []

    api = AgenticSystemClient(base_url=f"http://{host}:{port}")

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
    ] + [t.get_tool_definition() for t in custom_tools]

    if not disable_safety:
        for t in tool_definitions:
            t.input_shields = [ShieldDefinition(shield_type=BuiltinShield.llama_guard)]
            t.output_shields = [
                ShieldDefinition(shield_type=BuiltinShield.llama_guard),
                ShieldDefinition(shield_type=BuiltinShield.injection_shield),
            ]

    create_request = AgenticSystemCreateRequest(
        model=model,
        instance_config=AgenticSystemInstanceConfig(
            instructions="You are a helpful assistant",
            available_tools=tool_definitions,
            input_shields=(
                []
                if disable_safety
                else [
                    ShieldDefinition(shield_type=BuiltinShield.llama_guard),
                    ShieldDefinition(shield_type=BuiltinShield.jailbreak_shield),
                ]
            ),
            output_shields=(
                []
                if disable_safety
                else [
                    ShieldDefinition(shield_type=BuiltinShield.llama_guard),
                ]
            ),
            sampling_params=SamplingParams(),
        ),
    )
    create_response = await api.create_agentic_system(create_request)
    return AgenticSystemClientWrapper(api, create_response.system_id, custom_tools)
