# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import os
import uuid
from typing import AsyncGenerator

from llama_toolchain.inference.api import Inference
from llama_toolchain.memory.api import Memory
from llama_toolchain.safety.api import Safety
from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.tools.builtin import (
    BraveSearchTool,
    CodeInterpreterTool,
    PhotogenTool,
    WolframAlphaTool,
)
from llama_toolchain.tools.safety import with_safety

from .agent_instance import ChatAgent
from .config import MetaReferenceImplConfig


logger = logging.getLogger()
logger.setLevel(logging.INFO)


AGENT_INSTANCES_BY_ID = {}


class MetaReferenceAgenticSystemImpl(AgenticSystem):
    def __init__(
        self,
        config: MetaReferenceImplConfig,
        inference_api: Inference,
        memory_api: Memory,
        safety_api: Safety,
    ):
        self.config = config
        self.inference_api = inference_api
        self.memory_api = memory_api
        self.safety_api = safety_api

    async def initialize(self) -> None:
        pass

    async def create_agentic_system(
        self,
        agent_config: AgentConfig,
    ) -> AgenticSystemCreateResponse:
        agent_id = str(uuid.uuid4())

        builtin_tools = []
        for tool_defn in agent_config.tools:
            if isinstance(tool_defn, WolframAlphaToolDefinition):
                key = self.config.wolfram_api_key
                if not key:
                    raise ValueError("Wolfram API key not defined in config")
                tool = WolframAlphaTool(key)
            elif isinstance(tool_defn, BraveSearchToolDefinition):
                key = self.config.brave_search_api_key
                if not key:
                    raise ValueError("Brave API key not defined in config")
                tool = BraveSearchTool(key)
            elif isinstance(tool_defn, CodeInterpreterToolDefinition):
                tool = CodeInterpreterTool()
            elif isinstance(tool_defn, PhotogenToolDefinition):
                tool = PhotogenTool(
                    dump_dir="/tmp/photogen_dump_" + os.environ["USER"],
                )
            else:
                continue

            builtin_tools.append(
                with_safety(
                    tool,
                    self.safety_api,
                    tool_defn.input_shields,
                    tool_defn.output_shields,
                )
            )

        AGENT_INSTANCES_BY_ID[agent_id] = ChatAgent(
            agent_config=agent_config,
            inference_api=self.inference_api,
            safety_api=self.safety_api,
            memory_api=self.memory_api,
            builtin_tools=builtin_tools,
        )

        return AgenticSystemCreateResponse(
            agent_id=agent_id,
        )

    async def create_agentic_system_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgenticSystemSessionCreateResponse:
        assert agent_id in AGENT_INSTANCES_BY_ID, f"System {agent_id} not found"
        agent = AGENT_INSTANCES_BY_ID[agent_id]

        session = agent.create_session(session_name)
        return AgenticSystemSessionCreateResponse(
            session_id=session.session_id,
        )

    async def create_agentic_system_turn(
        self,
        request: AgenticSystemTurnCreateRequest,
    ) -> AsyncGenerator:
        agent_id = request.agent_id
        assert agent_id in AGENT_INSTANCES_BY_ID, f"System {agent_id} not found"
        agent = AGENT_INSTANCES_BY_ID[agent_id]

        assert (
            request.session_id in agent.sessions
        ), f"Session {request.session_id} not found"
        async for event in agent.create_and_execute_turn(request):
            yield event
