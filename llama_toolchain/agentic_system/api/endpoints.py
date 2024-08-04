# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .datatypes import *  # noqa: F403
from typing import Protocol

# this dependency is annoying and we need a forked up version anyway
from pyopenapi import webmethod

from strong_typing.schema import json_schema_type


@json_schema_type
class AgenticSystemCreateRequest(BaseModel):
    model: str
    instance_config: AgenticSystemInstanceConfig


@json_schema_type
class AgenticSystemCreateResponse(BaseModel):
    system_id: str


@json_schema_type
class AgenticSystemSessionCreateRequest(BaseModel):
    system_id: str
    session_name: str


@json_schema_type
class AgenticSystemSessionCreateResponse(BaseModel):
    session_id: str


@json_schema_type
# what's the URI?
class AgenticSystemTurnCreateRequest(BaseModel):
    system_id: str
    session_id: str

    messages: List[
        Union[
            UserMessage,
            ToolResponseMessage,
        ]
    ]

    stream: Optional[bool] = False
    override_config: Optional[AgenticSystemInstanceConfig] = None


@json_schema_type(
    schema={"description": "Server side event (SSE) stream of these events"}
)
class AgenticSystemTurnResponseStreamChunk(BaseModel):
    event: AgenticSystemTurnResponseEvent


@json_schema_type
class AgenticSystemStepResponse(BaseModel):
    step: Step


class AgenticSystem(Protocol):

    @webmethod(route="/agentic_system/create")
    async def create_agentic_system(
        self,
        request: AgenticSystemCreateRequest,
    ) -> AgenticSystemCreateResponse: ...

    @webmethod(route="/agentic_system/turn/create")
    async def create_agentic_system_turn(
        self,
        request: AgenticSystemTurnCreateRequest,
    ) -> AgenticSystemTurnResponseStreamChunk: ...

    @webmethod(route="/agentic_system/turn/get")
    async def get_agentic_system_turn(
        self,
        agent_id: str,
        turn_id: str,
    ) -> Turn: ...

    @webmethod(route="/agentic_system/step/get")
    async def get_agentic_system_step(
        self, agent_id: str, turn_id: str, step_id: str
    ) -> AgenticSystemStepResponse: ...

    @webmethod(route="/agentic_system/session/create")
    async def create_agentic_system_session(
        self,
        request: AgenticSystemSessionCreateRequest,
    ) -> AgenticSystemSessionCreateResponse: ...

    @webmethod(route="/agentic_system/memory_bank/attach")
    async def attach_memory_bank_to_agentic_system(
        self,
        agent_id: str,
        session_id: str,
        memory_bank_ids: List[str],
    ) -> None: ...

    @webmethod(route="/agentic_system/memory_bank/detach")
    async def detach_memory_bank_from_agentic_system(
        self,
        agent_id: str,
        session_id: str,
        memory_bank_ids: List[str],
    ) -> None: ...

    @webmethod(route="/agentic_system/session/get")
    async def get_agentic_system_session(
        self,
        agent_id: str,
        session_id: str,
        turn_ids: Optional[List[str]] = None,
    ) -> Session: ...

    @webmethod(route="/agentic_system/session/delete")
    async def delete_agentic_system_session(
        self, agent_id: str, session_id: str
    ) -> None: ...

    @webmethod(route="/agentic_system/delete")
    async def delete_agentic_system(
        self,
        agent_id: str,
    ) -> None: ...
