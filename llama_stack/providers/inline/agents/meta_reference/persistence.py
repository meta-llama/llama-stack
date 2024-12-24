# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel

from llama_stack.apis.agents import *  # noqa: F403
from llama_stack.providers.utils.kvstore import KVStore

log = logging.getLogger(__name__)


class AgentSessionInfo(BaseModel):
    session_id: str
    session_name: str
    started_at: datetime


class AgentPersistence:
    def __init__(self, agent_id: str, kvstore: KVStore):
        self.agent_id = agent_id
        self.kvstore = kvstore

    async def create_session(self, name: str) -> str:
        session_id = str(uuid.uuid4())
        session_info = AgentSessionInfo(
            session_id=session_id,
            session_name=name,
            started_at=datetime.now(),
        )
        await self.kvstore.set(
            key=f"session:{self.agent_id}:{session_id}",
            value=session_info.model_dump_json(),
        )
        return session_id

    async def get_session_info(self, session_id: str) -> Optional[AgentSessionInfo]:
        value = await self.kvstore.get(
            key=f"session:{self.agent_id}:{session_id}",
        )
        if not value:
            return None

        return AgentSessionInfo(**json.loads(value))

    async def add_turn_to_session(self, session_id: str, turn: Turn):
        await self.kvstore.set(
            key=f"session:{self.agent_id}:{session_id}:{turn.turn_id}",
            value=turn.model_dump_json(),
        )

    async def get_session_turns(self, session_id: str) -> List[Turn]:
        values = await self.kvstore.range(
            start_key=f"session:{self.agent_id}:{session_id}:",
            end_key=f"session:{self.agent_id}:{session_id}:\xff\xff\xff\xff",
        )
        turns = []
        for value in values:
            try:
                turn = Turn(**json.loads(value))
                turns.append(turn)
            except Exception as e:
                log.error(f"Error parsing turn: {e}")
                continue
        turns.sort(key=lambda x: (x.completed_at or datetime.min))
        return turns
