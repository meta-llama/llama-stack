# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel

from llama_stack.apis.agents import ToolExecutionStep, Turn
from llama_stack.distribution.access_control import check_access
from llama_stack.distribution.datatypes import AccessAttributes
from llama_stack.distribution.request_headers import get_auth_attributes
from llama_stack.providers.utils.kvstore import KVStore

log = logging.getLogger(__name__)


class AgentSessionInfo(BaseModel):
    session_id: str
    session_name: str
    # TODO: is this used anywhere?
    vector_db_id: Optional[str] = None
    started_at: datetime
    access_attributes: Optional[AccessAttributes] = None


class AgentPersistence:
    def __init__(self, agent_id: str, kvstore: KVStore):
        self.agent_id = agent_id
        self.kvstore = kvstore

    async def create_session(self, name: str) -> str:
        session_id = str(uuid.uuid4())

        # Get current user's auth attributes for new sessions
        auth_attributes = get_auth_attributes()
        access_attributes = AccessAttributes(**auth_attributes) if auth_attributes else None

        session_info = AgentSessionInfo(
            session_id=session_id,
            session_name=name,
            started_at=datetime.now(timezone.utc),
            access_attributes=access_attributes,
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

        session_info = AgentSessionInfo(**json.loads(value))

        # Check access to session
        if not self._check_session_access(session_info):
            return None

        return session_info

    def _check_session_access(self, session_info: AgentSessionInfo) -> bool:
        """Check if current user has access to the session."""
        # Handle backward compatibility for old sessions without access control
        if not hasattr(session_info, "access_attributes"):
            return True

        return check_access(session_info.session_id, session_info.access_attributes, get_auth_attributes())

    async def get_session_if_accessible(self, session_id: str) -> Optional[AgentSessionInfo]:
        """Get session info if the user has access to it. For internal use by sub-session methods."""
        session_info = await self.get_session_info(session_id)
        if not session_info:
            return None

        return session_info

    async def add_vector_db_to_session(self, session_id: str, vector_db_id: str):
        session_info = await self.get_session_if_accessible(session_id)
        if session_info is None:
            raise ValueError(f"Session {session_id} not found or access denied")

        session_info.vector_db_id = vector_db_id
        await self.kvstore.set(
            key=f"session:{self.agent_id}:{session_id}",
            value=session_info.model_dump_json(),
        )

    async def add_turn_to_session(self, session_id: str, turn: Turn):
        if not await self.get_session_if_accessible(session_id):
            raise ValueError(f"Session {session_id} not found or access denied")

        await self.kvstore.set(
            key=f"session:{self.agent_id}:{session_id}:{turn.turn_id}",
            value=turn.model_dump_json(),
        )

    async def get_session_turns(self, session_id: str) -> List[Turn]:
        if not await self.get_session_if_accessible(session_id):
            raise ValueError(f"Session {session_id} not found or access denied")

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

    async def get_session_turn(self, session_id: str, turn_id: str) -> Optional[Turn]:
        if not await self.get_session_if_accessible(session_id):
            raise ValueError(f"Session {session_id} not found or access denied")

        value = await self.kvstore.get(
            key=f"session:{self.agent_id}:{session_id}:{turn_id}",
        )
        if not value:
            return None
        return Turn(**json.loads(value))

    async def set_in_progress_tool_call_step(self, session_id: str, turn_id: str, step: ToolExecutionStep):
        if not await self.get_session_if_accessible(session_id):
            raise ValueError(f"Session {session_id} not found or access denied")

        await self.kvstore.set(
            key=f"in_progress_tool_call_step:{self.agent_id}:{session_id}:{turn_id}",
            value=step.model_dump_json(),
        )

    async def get_in_progress_tool_call_step(self, session_id: str, turn_id: str) -> Optional[ToolExecutionStep]:
        if not await self.get_session_if_accessible(session_id):
            return None

        value = await self.kvstore.get(
            key=f"in_progress_tool_call_step:{self.agent_id}:{session_id}:{turn_id}",
        )
        return ToolExecutionStep(**json.loads(value)) if value else None

    async def set_num_infer_iters_in_turn(self, session_id: str, turn_id: str, num_infer_iters: int):
        if not await self.get_session_if_accessible(session_id):
            raise ValueError(f"Session {session_id} not found or access denied")

        await self.kvstore.set(
            key=f"num_infer_iters_in_turn:{self.agent_id}:{session_id}:{turn_id}",
            value=str(num_infer_iters),
        )

    async def get_num_infer_iters_in_turn(self, session_id: str, turn_id: str) -> Optional[int]:
        if not await self.get_session_if_accessible(session_id):
            return None

        value = await self.kvstore.get(
            key=f"num_infer_iters_in_turn:{self.agent_id}:{session_id}:{turn_id}",
        )
        return int(value) if value else None
