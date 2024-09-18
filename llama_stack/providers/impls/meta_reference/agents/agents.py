# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import logging
import tempfile
import uuid
from typing import AsyncGenerator

from llama_stack.apis.inference import Inference
from llama_stack.apis.memory import Memory
from llama_stack.apis.safety import Safety
from llama_stack.apis.agents import *  # noqa: F403

from .agent_instance import ChatAgent
from .config import MetaReferenceImplConfig
from .tools.builtin import (
    CodeInterpreterTool,
    PhotogenTool,
    SearchTool,
    WolframAlphaTool,
)
from .tools.safety import with_safety


logger = logging.getLogger()
logger.setLevel(logging.INFO)


AGENT_INSTANCES_BY_ID = {}


class MetaReferenceAgentsImpl(Agents):
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

    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        agent_id = str(uuid.uuid4())

        builtin_tools = []
        for tool_defn in agent_config.tools:
            if isinstance(tool_defn, WolframAlphaToolDefinition):
                tool = WolframAlphaTool(tool_defn.api_key)
            elif isinstance(tool_defn, SearchToolDefinition):
                tool = SearchTool(tool_defn.engine, tool_defn.api_key)
            elif isinstance(tool_defn, CodeInterpreterToolDefinition):
                tool = CodeInterpreterTool()
            elif isinstance(tool_defn, PhotogenToolDefinition):
                tool = PhotogenTool(dump_dir=tempfile.mkdtemp())
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

        return AgentCreateResponse(
            agent_id=agent_id,
        )

    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        assert agent_id in AGENT_INSTANCES_BY_ID, f"System {agent_id} not found"
        agent = AGENT_INSTANCES_BY_ID[agent_id]

        session = agent.create_session(session_name)
        return AgentSessionCreateResponse(
            session_id=session.session_id,
        )

    async def create_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: List[
            Union[
                UserMessage,
                ToolResponseMessage,
            ]
        ],
        attachments: Optional[List[Attachment]] = None,
        stream: Optional[bool] = False,
    ) -> AsyncGenerator:
        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            attachments=attachments,
            stream=stream,
        )

        agent_id = request.agent_id
        assert agent_id in AGENT_INSTANCES_BY_ID, f"System {agent_id} not found"
        agent = AGENT_INSTANCES_BY_ID[agent_id]

        assert (
            request.session_id in agent.sessions
        ), f"Session {request.session_id} not found"
        async for event in agent.create_and_execute_turn(request):
            yield event
