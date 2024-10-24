# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
import uuid
from typing import AsyncGenerator

from llama_stack.apis.inference import Inference
from llama_stack.apis.memory import Memory
from llama_stack.apis.memory_banks import MemoryBanks
from llama_stack.apis.safety import Safety
from llama_stack.apis.agents import *  # noqa: F403

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
        memory_api: Memory,
        safety_api: Safety,
        memory_banks_api: MemoryBanks,
    ):
        self.config = config
        self.inference_api = inference_api
        self.memory_api = memory_api
        self.safety_api = safety_api
        self.memory_banks_api = memory_banks_api

        self.in_memory_store = InmemoryKVStoreImpl()

    async def initialize(self) -> None:
        self.persistence_store = await kvstore_impl(self.config.persistence_store)

    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        agent_id = str(uuid.uuid4())

        await self.persistence_store.set(
            key=f"agent:{agent_id}",
            value=agent_config.json(),
        )
        return AgentCreateResponse(
            agent_id=agent_id,
        )

    async def get_agent(self, agent_id: str) -> ChatAgent:
        agent_config = await self.persistence_store.get(
            key=f"agent:{agent_id}",
        )
        if not agent_config:
            raise ValueError(f"Could not find agent config for {agent_id}")

        try:
            agent_config = json.loads(agent_config)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Could not JSON decode agent config for {agent_id}"
            ) from e

        try:
            agent_config = AgentConfig(**agent_config)
        except Exception as e:
            raise ValueError(
                f"Could not validate(?) agent config for {agent_id}"
            ) from e

        return ChatAgent(
            agent_id=agent_id,
            agent_config=agent_config,
            inference_api=self.inference_api,
            safety_api=self.safety_api,
            memory_api=self.memory_api,
            memory_banks_api=self.memory_banks_api,
            persistence_store=(
                self.persistence_store
                if agent_config.enable_session_persistence
                else self.in_memory_store
            ),
        )

    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        agent = await self.get_agent(agent_id)

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
        attachments: Optional[List[Attachment]] = None,
        stream: Optional[bool] = False,
    ) -> AsyncGenerator:
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            attachments=attachments,
            stream=True,
        )
        if stream:
            return self._create_agent_turn_streaming(request)
        else:
            raise NotImplementedError("Non-streaming agent turns not yet implemented")

    async def _create_agent_turn_streaming(
        self,
        request: AgentTurnCreateRequest,
    ) -> AsyncGenerator:
        agent = await self.get_agent(request.agent_id)
        async for event in agent.create_and_execute_turn(request):
            yield event

    async def get_agents_turn(
        self, agent_id: str, session_id: str, turn_id: str
    ) -> Turn:
        turn = await self.persistence_store.get(
            f"session:{agent_id}:{session_id}:{turn_id}"
        )
        turn = json.loads(turn)
        turn = Turn(**turn)
        return turn

    async def get_agents_step(
        self, agent_id: str, session_id: str, turn_id: str, step_id: str
    ) -> AgentStepResponse:
        turn = await self.persistence_store.get(
            f"session:{agent_id}:{session_id}:{turn_id}"
        )
        turn = json.loads(turn)
        turn = Turn(**turn)
        steps = turn.steps
        for step in steps:
            if step.step_id == step_id:
                return AgentStepResponse(step=step)
        raise ValueError(f"Provided step_id {step_id} could not be found")

    async def get_agents_session(
        self,
        agent_id: str,
        session_id: str,
        turn_ids: Optional[List[str]] = None,
    ) -> Session:
        session = await self.persistence_store.get(f"session:{agent_id}:{session_id}")
        session = Session(**json.loads(session), turns=[])
        turns = []
        if turn_ids:
            for turn_id in turn_ids:
                turn = await self.persistence_store.get(
                    f"session:{agent_id}:{session_id}:{turn_id}"
                )
                turn = json.loads(turn)
                turn = Turn(**turn)
                turns.append(turn)
        return Session(
            session_name=session.session_name,
            session_id=session_id,
            turns=turns if turns else [],
            started_at=session.started_at,
        )

    async def delete_agents_session(self, agent_id: str, session_id: str) -> None:
        await self.persistence_store.delete(f"session:{agent_id}:{session_id}")

    async def delete_agents(self, agent_id: str) -> None:
        await self.persistence_store.delete(f"agent:{agent_id}")
