# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import shutil
import uuid
from datetime import datetime, timezone
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
    Session,
    Turn,
)
from llama_stack.apis.common.responses import PaginatedResponse
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
from llama_stack.providers.utils.datasetio.pagination import paginate_records
from llama_stack.providers.utils.kvstore import InmemoryKVStoreImpl, kvstore_impl

from .agent_instance import ChatAgent
from .config import MetaReferenceAgentsImplConfig
from .persistence import AgentInfo

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
        created_at = datetime.now(timezone.utc)

        # Create AgentInfo with the config and created_at
        agent_info = AgentInfo(
            **agent_config.model_dump(),
            created_at=created_at,
        )

        # Store the agent info
        await self.persistence_store.set(
            key=f"agent:{agent_id}",
            value=agent_info.model_dump_json(),
        )

        return AgentCreateResponse(
            agent_id=agent_id,
        )

    async def _get_agent_impl(self, agent_id: str) -> ChatAgent:
        agent_info = await self.persistence_store.get(
            key=f"agent:{agent_id}",
        )
        if not agent_info:
            raise ValueError(f"Could not find agent info for {agent_id}")

        try:
            agent_info = AgentInfo.model_validate_json(agent_info)
        except Exception as e:
            raise ValueError(f"Could not validate agent info for {agent_id}") from e

        return ChatAgent(
            agent_id=agent_id,
            agent_config=agent_info,
            inference_api=self.inference_api,
            safety_api=self.safety_api,
            vector_io_api=self.vector_io_api,
            tool_runtime_api=self.tool_runtime_api,
            tool_groups_api=self.tool_groups_api,
            persistence_store=(
                self.persistence_store if agent_info.enable_session_persistence else self.in_memory_store
            ),
            created_at=agent_info.created_at,
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
        agent = await self._get_agent_impl(agent_id)
        session_info = await agent.storage.get_session_info(session_id)
        if session_info is None:
            raise ValueError(f"Session {session_id} not found")

        # Delete turns first, then the session
        await agent.storage.delete_session_turns(session_id)
        await agent.storage.delete_session(session_id)

    async def delete_agent(self, agent_id: str) -> None:
        # First get all sessions for this agent
        agent = await self._get_agent_impl(agent_id)
        sessions = await agent.storage.list_sessions()

        # Delete all sessions
        for session in sessions:
            await self.delete_agents_session(agent_id, session.session_id)

        # Finally delete the agent itself
        await self.persistence_store.delete(f"agent:{agent_id}")

    async def list_agents(self, start_index: int | None = None, limit: int | None = None) -> PaginatedResponse:
        agent_keys = await self.persistence_store.range_key("agent:", "agent:\xff")
        agent_list: List[Agent] = []
        for agent_key in agent_keys:
            agent_id = agent_key.split(":")[1]

            # Get the agent info using the key
            agent_info_json = await self.persistence_store.get(agent_key)
            if not agent_info_json:
                logger.error(f"Could not find agent info for key {agent_key}")
                continue

            try:
                agent_info = AgentInfo.model_validate_json(agent_info_json)
                agent_list.append(
                    Agent(
                        agent_id=agent_id,
                        agent_config=agent_info,
                        created_at=agent_info.created_at,
                    )
                )
            except Exception as e:
                logger.error(f"Error parsing agent info for {agent_id}: {e}")
                continue

        # Convert Agent objects to dictionaries
        agent_dicts = [agent.model_dump() for agent in agent_list]
        return paginate_records(agent_dicts, start_index, limit)

    async def get_agent(self, agent_id: str) -> Agent:
        chat_agent = await self._get_agent_impl(agent_id)
        agent = Agent(
            agent_id=agent_id,
            agent_config=chat_agent.agent_config,
            created_at=chat_agent.created_at,
        )
        return agent

    async def list_agent_sessions(
        self, agent_id: str, start_index: int | None = None, limit: int | None = None
    ) -> PaginatedResponse:
        agent = await self._get_agent_impl(agent_id)
        sessions = await agent.storage.list_sessions()
        # Convert Session objects to dictionaries
        session_dicts = [session.model_dump() for session in sessions]
        return paginate_records(session_dicts, start_index, limit)

    async def shutdown(self) -> None:
        pass
