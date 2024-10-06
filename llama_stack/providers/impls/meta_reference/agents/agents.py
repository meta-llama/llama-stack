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
    ):
        self.config = config
        self.inference_api = inference_api
        self.memory_api = memory_api
        self.safety_api = safety_api
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
        agent = await self.get_agent(agent_id)

        # wrapper request to make it easier to pass around (internal only, not exposed to API)
        request = AgentTurnCreateRequest(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            attachments=attachments,
            stream=stream,
        )

        async for event in agent.create_and_execute_turn(request):
            yield event
