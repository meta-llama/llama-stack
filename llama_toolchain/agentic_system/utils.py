# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.agentic_system.client import AgenticSystemClient
from llama_toolchain.memory.api import *  # noqa: F403
from llama_toolchain.safety.api import *  # noqa: F403
from llama_toolchain.tools.custom.datatypes import CustomTool

from .execute_with_custom_tools import AgentWithCustomToolExecutor


class AttachmentBehavior(Enum):
    rag = "rag"
    code_interpreter = "code_interpreter"
    auto = "auto"


def default_builtins() -> List[BuiltinTool]:
    return [
        BuiltinTool.brave_search,
        BuiltinTool.wolfram_alpha,
        BuiltinTool.photogen,
        BuiltinTool.code_interpreter,
    ]


@dataclass
class QuickToolConfig:
    custom_tools: List[CustomTool] = field(default_factory=list)

    prompt_format: ToolPromptFormat = ToolPromptFormat.json

    # use this to control whether you want the model to write code to
    # process them, or you want to "RAG" them beforehand
    attachment_behavior: AttachmentBehavior = AttachmentBehavior.auto

    builtin_tools: List[BuiltinTool] = field(default_factory=default_builtins)

    # if you have a memory bank already pre-populated, specify it here
    memory_bank_id: Optional[str] = None


# This is a utility function; it does not provide all bells and whistles
# you can get from the underlying AgenticSystem API. Any limitations should
# ideally be resolved by making another well-scoped utility function instead
# of adding complex options here.
async def make_agent_config_with_custom_tools(
    model: str = "Meta-Llama3.1-8B-Instruct",
    disable_safety: bool = False,
    tool_config: QuickToolConfig = QuickToolConfig(),
) -> AgentConfig:
    tool_definitions = []

    # ensure code interpreter is enabled if attachments need it
    builtin_tools = tool_config.builtin_tools
    tool_choice = ToolChoice.auto
    if tool_config.attachment_behavior == AttachmentBehavior.code_interpreter:
        if BuiltinTool.code_interpreter not in builtin_tools:
            builtin_tools.append(BuiltinTool.code_interpreter)

        tool_choice = ToolChoice.required

    for t in builtin_tools:
        if t == BuiltinTool.brave_search:
            tool_definitions.append(BraveSearchToolDefinition())
        elif t == BuiltinTool.wolfram_alpha:
            tool_definitions.append(WolframAlphaToolDefinition())
        elif t == BuiltinTool.photogen:
            tool_definitions.append(PhotogenToolDefinition())
        elif t == BuiltinTool.code_interpreter:
            tool_definitions.append(CodeInterpreterToolDefinition())

    # enable memory unless we are specifically disabling it
    if tool_config.attachment_behavior != AttachmentBehavior.code_interpreter:
        bank_configs = []
        if tool_config.memory_bank_id:
            bank_configs.append(
                AgenticSystemVectorMemoryBankConfig(bank_id=tool_config.memory_bank_id)
            )
        tool_definitions.append(MemoryToolDefinition(memory_bank_configs=bank_configs))

    tool_definitions += [t.get_tool_definition() for t in tool_config.custom_tools]

    if not disable_safety:
        for t in tool_definitions:
            t.input_shields = [ShieldDefinition(shield_type=BuiltinShield.llama_guard)]
            t.output_shields = [
                ShieldDefinition(shield_type=BuiltinShield.llama_guard),
                ShieldDefinition(shield_type=BuiltinShield.injection_shield),
            ]

    cfg = AgentConfig(
        model=model,
        instructions="You are a helpful assistant",
        sampling_params=SamplingParams(),
        tools=tool_definitions,
        tool_prompt_format=tool_config.prompt_format,
        tool_choice=tool_choice,
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
    )
    return cfg


async def get_agent_with_custom_tools(
    host: str,
    port: int,
    agent_config: AgentConfig,
    custom_tools: List[CustomTool],
) -> AgentWithCustomToolExecutor:
    api = AgenticSystemClient(base_url=f"http://{host}:{port}")

    create_response = await api.create_agentic_system(agent_config)
    agent_id = create_response.agent_id

    name = f"Session-{uuid.uuid4()}"
    response = await api.create_agentic_system_session(
        agent_id=agent_id,
        session_name=name,
    )
    session_id = response.session_id

    return AgentWithCustomToolExecutor(
        api, agent_id, session_id, agent_config, custom_tools
    )
