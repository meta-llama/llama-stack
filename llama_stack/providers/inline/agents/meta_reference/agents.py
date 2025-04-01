# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
import shutil
import tempfile
import uuid
from typing import AsyncGenerator, List, Optional, Union

from llama_stack.apis.agents import (
    Agent,
    AgentConfig,
    AgentCreateResponse,
    Agents,
    AgentSessionCreateResponse,
    AgentStepResponse,
    AgentToolGroup,
    AgentTurnCreateRequest,
    AgentTurnResumeRequest,
    Document,
    ListAgentSessionsResponse,
    ListAgentsResponse,
    Session,
    Turn,
)
from llama_stack.apis.inference import (
    Inference,
    ToolConfig,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import Safety
from llama_stack.apis.tools import ToolGroups, ToolRuntime
from llama_stack.apis.vector_io import VectorIO
from llama_stack.providers.utils.kvstore import InmemoryKVStoreImpl, kvstore_impl

from .agent_instance import ChatAgent
from .config import MetaReferenceAgentsImplConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MetaReferenceAgentsImpl(Agents):
    def __init__(
        self,
        config: MetaReferenceAgentsImplConfig,
        inference_api: Inference,
        vector_io_api: VectorIO,
        safety_api: Safety,
        tool_runtime_api: ToolRuntime,
        tool_groups_api: ToolGroups,
    ):
        self.config = config
        self.inference_api = inference_api
        self.vector_io_api = vector_io_api
        self.safety_api = safety_api
        self.tool_runtime_api = tool_runtime_api
        self.tool_groups_api = tool_groups_api

        self.in_memory_store = InmemoryKVStoreImpl()
        self.tempdir = tempfile.mkdtemp()

    async def initialize(self) -> None:
        self.persistence_store = await kvstore_impl(self.config.persistence_store)

        # check if "bwrap" is available
        if not shutil.which("bwrap"):
            logger.warning("Warning: `bwrap` is not available. Code interpreter tool will not work correctly.")

    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        agent_id = str(uuid.uuid4())

        await self.persistence_store.set(
            key=f"agent:{agent_id}",
            value=agent_config.model_dump_json(),
        )
        return AgentCreateResponse(
            agent_id=agent_id,
        )

    async def _get_agent_impl(self, agent_id: str) -> ChatAgent:
        agent_config = await self.persistence_store.get(
            key=f"agent:{agent_id}",
        )
        if not agent_config:
            raise ValueError(f"Could not find agent config for {agent_id}")

        try:
            agent_config = json.loads(agent_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not JSON decode agent config for {agent_id}") from e

        try:
            agent_config = AgentConfig(**agent_config)
        except Exception as e:
            raise ValueError(f"Could not validate(?) agent config for {agent_id}") from e

        return ChatAgent(
            agent_id=agent_id,
            agent_config=agent_config,
            tempdir=self.tempdir,
            inference_api=self.inference_api,
            safety_api=self.safety_api,
            vector_io_api=self.vector_io_api,
            tool_runtime_api=self.tool_runtime_api,
            tool_groups_api=self.tool_groups_api,
            persistence_store=(
                self.persistence_store if agent_config.enable_session_persistence else self.in_memory_store
            ),
        )

    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        agent = await self._get_agent_impl(agent_id)

        session_id = await agent.create_session(session_name)
        return AgentSessionCreateResponse(
            session_id=session_id,
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
        toolgroups: Optional[List[AgentToolGroup]] = None,
        documents: Optional[List[Document]] = None,
        stream: Optional[bool] = False,
        tool_config: Optional[ToolConfig] = None,
    ) -> AsyncGenerator:
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            stream=True,
            toolgroups=toolgroups,
            documents=documents,
            tool_config=tool_config,
        )
        if stream:
            return self._create_agent_turn_streaming(request)
        else:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")

    async def _create_agent_turn_streaming(
        self,
        request: AgentTurnCreateRequest,
    ) -> AsyncGenerator:
        agent = await self._get_agent_impl(request.agent_id)
        async for event in agent.create_and_execute_turn(request):
            yield event

    async def resume_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        tool_responses: List[ToolResponse],
        stream: Optional[bool] = False,
    ) -> AsyncGenerator:
        request = AgentTurnResumeRequest(
            agent_id=agent_id,
            session_id=session_id,
            turn_id=turn_id,
            tool_responses=tool_responses,
            stream=stream,
        )
        if stream:
            return self._continue_agent_turn_streaming(request)
        else:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")

    async def _continue_agent_turn_streaming(
        self,
        request: AgentTurnResumeRequest,
    ) -> AsyncGenerator:
        agent = await self._get_agent_impl(request.agent_id)
        async for event in agent.resume_turn(request):
            yield event

    async def get_agents_turn(self, agent_id: str, session_id: str, turn_id: str) -> Turn:
        agent = await self._get_agent_impl(agent_id)
        turn = await agent.storage.get_session_turn(session_id, turn_id)
        return turn

    async def get_agents_step(self, agent_id: str, session_id: str, turn_id: str, step_id: str) -> AgentStepResponse:
        turn = await self.get_agents_turn(agent_id, session_id, turn_id)
        for step in turn.steps:
            if step.step_id == step_id:
                return AgentStepResponse(step=step)
        raise ValueError(f"Provided step_id {step_id} could not be found")

    async def get_agents_session(
        self,
        agent_id: str,
        session_id: str,
        turn_ids: Optional[List[str]] = None,
    ) -> Session:
        agent = await self._get_agent_impl(agent_id)
        session_info = await agent.storage.get_session_info(session_id)
        if session_info is None:
            raise ValueError(f"Session {session_id} not found")
        turns = await agent.storage.get_session_turns(session_id)
        if turn_ids:
            turns = [turn for turn in turns if turn.turn_id in turn_ids]
        return Session(
            session_name=session_info.session_name,
            session_id=session_id,
            turns=turns,
            started_at=session_info.started_at,
        )

    async def delete_agents_session(self, agent_id: str, session_id: str) -> None:
        await self.persistence_store.delete(f"session:{agent_id}:{session_id}")

    async def delete_agent(self, agent_id: str) -> None:
        await self.persistence_store.delete(f"agent:{agent_id}")

    async def shutdown(self) -> None:
        pass

    async def list_agents(self) -> ListAgentsResponse:
        pass

    async def get_agent(self, agent_id: str) -> Agent:
        pass

    async def list_agent_sessions(
        self,
        agent_id: str,
    ) -> ListAgentSessionsResponse:
        pass
