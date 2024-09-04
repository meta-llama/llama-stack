# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from llama_models.schema_utils import json_schema_type, webmethod

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_toolchain.common.deployment_types import *  # noqa: F403
from llama_toolchain.inference.api import *  # noqa: F403
from llama_toolchain.safety.api import *  # noqa: F403
from llama_toolchain.memory.api import *  # noqa: F403


@json_schema_type
class Attachment(BaseModel):
    content: InterleavedTextMedia | URL
    mime_type: str


class AgenticSystemTool(Enum):
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"

    function_call = "function_call"
    memory = "memory"


class ToolDefinitionCommon(BaseModel):
    input_shields: Optional[List[ShieldDefinition]] = Field(default_factory=list)
    output_shields: Optional[List[ShieldDefinition]] = Field(default_factory=list)


@json_schema_type
class BraveSearchToolDefinition(ToolDefinitionCommon):
    type: Literal[AgenticSystemTool.brave_search.value] = (
        AgenticSystemTool.brave_search.value
    )
    remote_execution: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class WolframAlphaToolDefinition(ToolDefinitionCommon):
    type: Literal[AgenticSystemTool.wolfram_alpha.value] = (
        AgenticSystemTool.wolfram_alpha.value
    )
    remote_execution: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class PhotogenToolDefinition(ToolDefinitionCommon):
    type: Literal[AgenticSystemTool.photogen.value] = AgenticSystemTool.photogen.value
    remote_execution: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class CodeInterpreterToolDefinition(ToolDefinitionCommon):
    type: Literal[AgenticSystemTool.code_interpreter.value] = (
        AgenticSystemTool.code_interpreter.value
    )
    enable_inline_code_execution: bool = True
    remote_execution: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class FunctionCallToolDefinition(ToolDefinitionCommon):
    type: Literal[AgenticSystemTool.function_call.value] = (
        AgenticSystemTool.function_call.value
    )
    function_name: str
    description: str
    parameters: Dict[str, ToolParamDefinition]
    remote_execution: Optional[RestAPIExecutionConfig] = None


class _MemoryBankConfigCommon(BaseModel):
    bank_id: str


class AgenticSystemVectorMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal[MemoryBankType.vector.value] = MemoryBankType.vector.value


class AgenticSystemKeyValueMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal[MemoryBankType.keyvalue.value] = MemoryBankType.keyvalue.value
    keys: List[str]  # what keys to focus on


class AgenticSystemKeywordMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal[MemoryBankType.keyword.value] = MemoryBankType.keyword.value


class AgenticSystemGraphMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal[MemoryBankType.graph.value] = MemoryBankType.graph.value
    entities: List[str]  # what entities to focus on


MemoryBankConfig = Annotated[
    Union[
        AgenticSystemVectorMemoryBankConfig,
        AgenticSystemKeyValueMemoryBankConfig,
        AgenticSystemKeywordMemoryBankConfig,
        AgenticSystemGraphMemoryBankConfig,
    ],
    Field(discriminator="type"),
]


@json_schema_type
class MemoryToolDefinition(ToolDefinitionCommon):
    type: Literal[AgenticSystemTool.memory.value] = AgenticSystemTool.memory.value
    memory_bank_configs: List[MemoryBankConfig] = Field(default_factory=list)
    max_tokens_in_context: int = 4096
    max_chunks: int = 10


AgenticSystemToolDefinition = Annotated[
    Union[
        BraveSearchToolDefinition,
        WolframAlphaToolDefinition,
        PhotogenToolDefinition,
        CodeInterpreterToolDefinition,
        FunctionCallToolDefinition,
        MemoryToolDefinition,
    ],
    Field(discriminator="type"),
]


class StepCommon(BaseModel):
    turn_id: str
    step_id: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class StepType(Enum):
    inference = "inference"
    tool_execution = "tool_execution"
    shield_call = "shield_call"
    memory_retrieval = "memory_retrieval"


@json_schema_type
class InferenceStep(StepCommon):
    model_config = ConfigDict(protected_namespaces=())

    step_type: Literal[StepType.inference.value] = StepType.inference.value
    model_response: CompletionMessage


@json_schema_type
class ToolExecutionStep(StepCommon):
    step_type: Literal[StepType.tool_execution.value] = StepType.tool_execution.value
    tool_calls: List[ToolCall]
    tool_responses: List[ToolResponse]


@json_schema_type
class ShieldCallStep(StepCommon):
    step_type: Literal[StepType.shield_call.value] = StepType.shield_call.value
    response: ShieldResponse


@json_schema_type
class MemoryRetrievalStep(StepCommon):
    step_type: Literal[StepType.memory_retrieval.value] = (
        StepType.memory_retrieval.value
    )
    memory_bank_ids: List[str]
    inserted_context: InterleavedTextMedia


Step = Annotated[
    Union[
        InferenceStep,
        ToolExecutionStep,
        ShieldCallStep,
        MemoryRetrievalStep,
    ],
    Field(discriminator="step_type"),
]


@json_schema_type
class Turn(BaseModel):
    """A single turn in an interaction with an Agentic System."""

    turn_id: str
    session_id: str
    input_messages: List[
        Union[
            UserMessage,
            ToolResponseMessage,
        ]
    ]
    steps: List[Step]
    output_message: CompletionMessage
    output_attachments: List[Attachment] = Field(default_factory=list)

    started_at: datetime
    completed_at: Optional[datetime] = None


@json_schema_type
class Session(BaseModel):
    """A single session of an interaction with an Agentic System."""

    session_id: str
    session_name: str
    turns: List[Turn]
    started_at: datetime

    memory_bank: Optional[MemoryBank] = None


class AgentConfigCommon(BaseModel):
    sampling_params: Optional[SamplingParams] = SamplingParams()

    input_shields: Optional[List[ShieldDefinition]] = Field(default_factory=list)
    output_shields: Optional[List[ShieldDefinition]] = Field(default_factory=list)

    tools: Optional[List[AgenticSystemToolDefinition]] = Field(default_factory=list)
    tool_choice: Optional[ToolChoice] = Field(default=ToolChoice.auto)
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )


@json_schema_type
class AgentConfig(AgentConfigCommon):
    model: str
    instructions: str


class AgentConfigOverridablePerTurn(AgentConfigCommon):
    instructions: Optional[str] = None


class AgenticSystemTurnResponseEventType(Enum):
    step_start = "step_start"
    step_complete = "step_complete"
    step_progress = "step_progress"

    turn_start = "turn_start"
    turn_complete = "turn_complete"


@json_schema_type
class AgenticSystemTurnResponseStepStartPayload(BaseModel):
    event_type: Literal[AgenticSystemTurnResponseEventType.step_start.value] = (
        AgenticSystemTurnResponseEventType.step_start.value
    )
    step_type: StepType
    step_id: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


@json_schema_type
class AgenticSystemTurnResponseStepCompletePayload(BaseModel):
    event_type: Literal[AgenticSystemTurnResponseEventType.step_complete.value] = (
        AgenticSystemTurnResponseEventType.step_complete.value
    )
    step_type: StepType
    step_details: Step


@json_schema_type
class AgenticSystemTurnResponseStepProgressPayload(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    event_type: Literal[AgenticSystemTurnResponseEventType.step_progress.value] = (
        AgenticSystemTurnResponseEventType.step_progress.value
    )
    step_type: StepType
    step_id: str

    model_response_text_delta: Optional[str] = None
    tool_call_delta: Optional[ToolCallDelta] = None
    tool_response_text_delta: Optional[str] = None


@json_schema_type
class AgenticSystemTurnResponseTurnStartPayload(BaseModel):
    event_type: Literal[AgenticSystemTurnResponseEventType.turn_start.value] = (
        AgenticSystemTurnResponseEventType.turn_start.value
    )
    turn_id: str


@json_schema_type
class AgenticSystemTurnResponseTurnCompletePayload(BaseModel):
    event_type: Literal[AgenticSystemTurnResponseEventType.turn_complete.value] = (
        AgenticSystemTurnResponseEventType.turn_complete.value
    )
    turn: Turn


@json_schema_type
class AgenticSystemTurnResponseEvent(BaseModel):
    """Streamed agent execution response."""

    payload: Annotated[
        Union[
            AgenticSystemTurnResponseStepStartPayload,
            AgenticSystemTurnResponseStepProgressPayload,
            AgenticSystemTurnResponseStepCompletePayload,
            AgenticSystemTurnResponseTurnStartPayload,
            AgenticSystemTurnResponseTurnCompletePayload,
        ],
        Field(discriminator="event_type"),
    ]


@json_schema_type
class AgenticSystemCreateResponse(BaseModel):
    agent_id: str


@json_schema_type
class AgenticSystemSessionCreateResponse(BaseModel):
    session_id: str


@json_schema_type
class AgenticSystemTurnCreateRequest(AgentConfigOverridablePerTurn):
    agent_id: str
    session_id: str

    # TODO: figure out how we can simplify this and make why
    # ToolResponseMessage needs to be here (it is function call
    # execution from outside the system)
    messages: List[
        Union[
            UserMessage,
            ToolResponseMessage,
        ]
    ]
    attachments: Optional[List[Attachment]] = None

    stream: Optional[bool] = False


@json_schema_type
class AgenticSystemTurnResponseStreamChunk(BaseModel):
    event: AgenticSystemTurnResponseEvent


@json_schema_type
class AgenticSystemStepResponse(BaseModel):
    step: Step


class AgenticSystem(Protocol):
    @webmethod(route="/agentic_system/create")
    async def create_agentic_system(
        self,
        agent_config: AgentConfig,
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
        agent_id: str,
        session_name: str,
    ) -> AgenticSystemSessionCreateResponse: ...

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
