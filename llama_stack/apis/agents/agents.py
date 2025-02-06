# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from datetime import datetime
from enum import Enum
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
    Union,
)

from llama_models.schema_utils import json_schema_type, register_schema, webmethod
from pydantic import BaseModel, ConfigDict, Field

from llama_stack.apis.common.content_types import ContentDelta, InterleavedContent, URL
from llama_stack.apis.inference import (
    CompletionMessage,
    ResponseFormat,
    SamplingParams,
    ToolCall,
    ToolChoice,
    ToolPromptFormat,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
    ToolConfig,
)
from llama_stack.apis.safety import SafetyViolation
from llama_stack.apis.tools import ToolDef
from llama_stack.providers.utils.telemetry.trace_protocol import trace_protocol


class Attachment(BaseModel):
    content: InterleavedContent | URL
    mime_type: str


class Document(BaseModel):
    content: InterleavedContent | URL
    mime_type: str


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
    violation: Optional[SafetyViolation]


@json_schema_type
class MemoryRetrievalStep(StepCommon):
    step_type: Literal[StepType.memory_retrieval.value] = StepType.memory_retrieval.value
    vector_db_ids: str
    inserted_context: InterleavedContent


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
    output_attachments: Optional[List[Attachment]] = Field(default_factory=list)

    started_at: datetime
    completed_at: Optional[datetime] = None


@json_schema_type
class Session(BaseModel):
    """A single session of an interaction with an Agentic System."""

    session_id: str
    session_name: str
    turns: List[Turn]
    started_at: datetime


class AgentToolGroupWithArgs(BaseModel):
    name: str
    args: Dict[str, Any]


AgentToolGroup = register_schema(
    Union[
        str,
        AgentToolGroupWithArgs,
    ],
    name="AgentTool",
)


class AgentConfigCommon(BaseModel):
    sampling_params: Optional[SamplingParams] = SamplingParams()

    input_shields: Optional[List[str]] = Field(default_factory=list)
    output_shields: Optional[List[str]] = Field(default_factory=list)
    toolgroups: Optional[List[AgentToolGroup]] = Field(default_factory=list)
    client_tools: Optional[List[ToolDef]] = Field(default_factory=list)
    tool_choice: Optional[ToolChoice] = Field(default=ToolChoice.auto, deprecated="use tool_config instead")
    tool_prompt_format: Optional[ToolPromptFormat] = Field(default=None, deprecated="use tool_config instead")
    tool_config: Optional[ToolConfig] = Field(default=None)

    max_infer_iters: Optional[int] = 10

    def model_post_init(self, __context):
        if self.tool_config:
            if self.tool_choice and self.tool_config.tool_choice != self.tool_choice:
                raise ValueError("tool_choice is deprecated. Use tool_choice in tool_config instead.")
            if self.tool_prompt_format and self.tool_config.tool_prompt_format != self.tool_prompt_format:
                raise ValueError("tool_prompt_format is deprecated. Use tool_prompt_format in tool_config instead.")
        if self.tool_config is None:
            self.tool_config = ToolConfig(
                tool_choice=self.tool_choice,
                tool_prompt_format=self.tool_prompt_format,
            )


@json_schema_type
class AgentConfig(AgentConfigCommon):
    model: str
    instructions: str
    enable_session_persistence: bool
    response_format: Optional[ResponseFormat] = None


class AgentConfigOverridablePerTurn(AgentConfigCommon):
    instructions: Optional[str] = None


class AgentTurnResponseEventType(Enum):
    step_start = "step_start"
    step_complete = "step_complete"
    step_progress = "step_progress"

    turn_start = "turn_start"
    turn_complete = "turn_complete"


@json_schema_type
class AgentTurnResponseStepStartPayload(BaseModel):
    event_type: Literal[AgentTurnResponseEventType.step_start.value] = AgentTurnResponseEventType.step_start.value
    step_type: StepType
    step_id: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


@json_schema_type
class AgentTurnResponseStepCompletePayload(BaseModel):
    event_type: Literal[AgentTurnResponseEventType.step_complete.value] = AgentTurnResponseEventType.step_complete.value
    step_type: StepType
    step_id: str
    step_details: Step


@json_schema_type
class AgentTurnResponseStepProgressPayload(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    event_type: Literal[AgentTurnResponseEventType.step_progress.value] = AgentTurnResponseEventType.step_progress.value
    step_type: StepType
    step_id: str

    delta: ContentDelta


@json_schema_type
class AgentTurnResponseTurnStartPayload(BaseModel):
    event_type: Literal[AgentTurnResponseEventType.turn_start.value] = AgentTurnResponseEventType.turn_start.value
    turn_id: str


@json_schema_type
class AgentTurnResponseTurnCompletePayload(BaseModel):
    event_type: Literal[AgentTurnResponseEventType.turn_complete.value] = AgentTurnResponseEventType.turn_complete.value
    turn: Turn


AgentTurnResponseEventPayload = register_schema(
    Annotated[
        Union[
            AgentTurnResponseStepStartPayload,
            AgentTurnResponseStepProgressPayload,
            AgentTurnResponseStepCompletePayload,
            AgentTurnResponseTurnStartPayload,
            AgentTurnResponseTurnCompletePayload,
        ],
        Field(discriminator="event_type"),
    ],
    name="AgentTurnResponseEventPayload",
)


@json_schema_type
class AgentTurnResponseEvent(BaseModel):
    payload: AgentTurnResponseEventPayload


@json_schema_type
class AgentCreateResponse(BaseModel):
    agent_id: str


@json_schema_type
class AgentSessionCreateResponse(BaseModel):
    session_id: str


@json_schema_type
class AgentTurnCreateRequest(AgentConfigOverridablePerTurn):
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

    documents: Optional[List[Document]] = None
    toolgroups: Optional[List[AgentToolGroup]] = None

    stream: Optional[bool] = False
    tool_config: Optional[ToolConfig] = None


@json_schema_type
class AgentTurnResponseStreamChunk(BaseModel):
    """streamed agent turn completion response."""

    event: AgentTurnResponseEvent


@json_schema_type
class AgentStepResponse(BaseModel):
    step: Step


@runtime_checkable
@trace_protocol
class Agents(Protocol):
    """Agents API for creating and interacting with agentic systems.

    Main functionalities provided by this API:
    - Create agents with specific instructions and ability to use tools.
    - Interactions with agents are grouped into sessions ("threads"), and each interaction is called a "turn".
    - Agents can be provided with various tools (see the ToolGroups and ToolRuntime APIs for more details).
    - Agents can be provided with various shields (see the Safety API for more details).
    - Agents can also use Memory to retrieve information from knowledge bases. See the RAG Tool and Vector IO APIs for more details.
    """

    @webmethod(route="/agents", method="POST")
    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse: ...

    @webmethod(route="/agents/{agent_id}/session/{session_id}/turn", method="POST")
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
        stream: Optional[bool] = False,
        documents: Optional[List[Document]] = None,
        toolgroups: Optional[List[AgentToolGroup]] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[Turn, AsyncIterator[AgentTurnResponseStreamChunk]]: ...

    @webmethod(route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}", method="GET")
    async def get_agents_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
    ) -> Turn: ...

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}/step/{step_id}",
        method="GET",
    )
    async def get_agents_step(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        step_id: str,
    ) -> AgentStepResponse: ...

    @webmethod(route="/agents/{agent_id}/session", method="POST")
    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse: ...

    @webmethod(route="/agents/{agent_id}/session/{session_id}", method="GET")
    async def get_agents_session(
        self,
        session_id: str,
        agent_id: str,
        turn_ids: Optional[List[str]] = None,
    ) -> Session: ...

    @webmethod(route="/agents/{agent_id}/session/{session_id}", method="DELETE")
    async def delete_agents_session(
        self,
        session_id: str,
        agent_id: str,
    ) -> None: ...

    @webmethod(route="/agents/{agent_id}", method="DELETE")
    async def delete_agent(
        self,
        agent_id: str,
    ) -> None: ...
