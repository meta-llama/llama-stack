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
    Union,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from llama_stack.apis.common.content_types import URL, ContentDelta, InterleavedContent
from llama_stack.apis.inference import (
    CompletionMessage,
    ResponseFormat,
    SamplingParams,
    ToolCall,
    ToolChoice,
    ToolConfig,
    ToolPromptFormat,
    ToolResponse,
    ToolResponseMessage,
    UserMessage,
)
from llama_stack.apis.safety import SafetyViolation
from llama_stack.apis.tools import ToolDef
from llama_stack.schema_utils import json_schema_type, register_schema, webmethod


class Attachment(BaseModel):
    """An attachment to an agent turn.

    :param content: The content of the attachment.
    :param mime_type: The MIME type of the attachment.
    """

    content: InterleavedContent | URL
    mime_type: str


class Document(BaseModel):
    """A document to be used by an agent.

    :param content: The content of the document.
    :param mime_type: The MIME type of the document.
    """

    content: InterleavedContent | URL
    mime_type: str


class StepCommon(BaseModel):
    """A common step in an agent turn.

    :param turn_id: The ID of the turn.
    :param step_id: The ID of the step.
    :param started_at: The time the step started.
    :param completed_at: The time the step completed.
    """

    turn_id: str
    step_id: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class StepType(Enum):
    """Type of the step in an agent turn.

    :cvar inference: The step is an inference step that calls an LLM.
    :cvar tool_execution: The step is a tool execution step that executes a tool call.
    :cvar shield_call: The step is a shield call step that checks for safety violations.
    :cvar memory_retrieval: The step is a memory retrieval step that retrieves context for vector dbs.
    """

    inference = "inference"
    tool_execution = "tool_execution"
    shield_call = "shield_call"
    memory_retrieval = "memory_retrieval"


@json_schema_type
class InferenceStep(StepCommon):
    """An inference step in an agent turn.

    :param model_response: The response from the LLM.
    """

    model_config = ConfigDict(protected_namespaces=())

    step_type: Literal[StepType.inference.value] = StepType.inference.value
    model_response: CompletionMessage


@json_schema_type
class ToolExecutionStep(StepCommon):
    """A tool execution step in an agent turn.

    :param tool_calls: The tool calls to execute.
    :param tool_responses: The tool responses from the tool calls.
    """

    step_type: Literal[StepType.tool_execution.value] = StepType.tool_execution.value
    tool_calls: List[ToolCall]
    tool_responses: List[ToolResponse]


@json_schema_type
class ShieldCallStep(StepCommon):
    """A shield call step in an agent turn.

    :param violation: The violation from the shield call.
    """

    step_type: Literal[StepType.shield_call.value] = StepType.shield_call.value
    violation: Optional[SafetyViolation]


@json_schema_type
class MemoryRetrievalStep(StepCommon):
    """A memory retrieval step in an agent turn.

    :param vector_db_ids: The IDs of the vector databases to retrieve context from.
    :param inserted_context: The context retrieved from the vector databases.
    """

    step_type: Literal[StepType.memory_retrieval.value] = StepType.memory_retrieval.value
    # TODO: should this be List[str]?
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


AgentToolGroup = Union[
    str,
    AgentToolGroupWithArgs,
]
register_schema(AgentToolGroup, name="AgentTool")


class AgentConfigCommon(BaseModel):
    sampling_params: Optional[SamplingParams] = Field(default_factory=SamplingParams)

    input_shields: Optional[List[str]] = Field(default_factory=list)
    output_shields: Optional[List[str]] = Field(default_factory=list)
    toolgroups: Optional[List[AgentToolGroup]] = Field(default_factory=list)
    client_tools: Optional[List[ToolDef]] = Field(default_factory=list)
    tool_choice: Optional[ToolChoice] = Field(default=None, deprecated="use tool_config instead")
    tool_prompt_format: Optional[ToolPromptFormat] = Field(default=None, deprecated="use tool_config instead")
    tool_config: Optional[ToolConfig] = Field(default=None)

    max_infer_iters: Optional[int] = 10

    def model_post_init(self, __context):
        if self.tool_config:
            if self.tool_choice and self.tool_config.tool_choice != self.tool_choice:
                raise ValueError("tool_choice is deprecated. Use tool_choice in tool_config instead.")
            if self.tool_prompt_format and self.tool_config.tool_prompt_format != self.tool_prompt_format:
                raise ValueError("tool_prompt_format is deprecated. Use tool_prompt_format in tool_config instead.")
        else:
            params = {}
            if self.tool_choice:
                params["tool_choice"] = self.tool_choice
            if self.tool_prompt_format:
                params["tool_prompt_format"] = self.tool_prompt_format
            self.tool_config = ToolConfig(**params)


@json_schema_type
class AgentConfig(AgentConfigCommon):
    model: str
    instructions: str
    enable_session_persistence: Optional[bool] = False
    response_format: Optional[ResponseFormat] = None


@json_schema_type
class Agent(BaseModel):
    agent_id: str
    agent_config: AgentConfig
    created_at: datetime


@json_schema_type
class ListAgentsResponse(BaseModel):
    data: List[Agent]


@json_schema_type
class ListAgentSessionsResponse(BaseModel):
    data: List[Session]


class AgentConfigOverridablePerTurn(AgentConfigCommon):
    instructions: Optional[str] = None


class AgentTurnResponseEventType(Enum):
    step_start = "step_start"
    step_complete = "step_complete"
    step_progress = "step_progress"

    turn_start = "turn_start"
    turn_complete = "turn_complete"
    turn_awaiting_input = "turn_awaiting_input"


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


@json_schema_type
class AgentTurnResponseTurnAwaitingInputPayload(BaseModel):
    event_type: Literal[AgentTurnResponseEventType.turn_awaiting_input.value] = (
        AgentTurnResponseEventType.turn_awaiting_input.value
    )
    turn: Turn


AgentTurnResponseEventPayload = Annotated[
    Union[
        AgentTurnResponseStepStartPayload,
        AgentTurnResponseStepProgressPayload,
        AgentTurnResponseStepCompletePayload,
        AgentTurnResponseTurnStartPayload,
        AgentTurnResponseTurnCompletePayload,
        AgentTurnResponseTurnAwaitingInputPayload,
    ],
    Field(discriminator="event_type"),
]
register_schema(AgentTurnResponseEventPayload, name="AgentTurnResponseEventPayload")


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
class AgentTurnResumeRequest(BaseModel):
    agent_id: str
    session_id: str
    turn_id: str
    tool_responses: List[ToolResponse]
    stream: Optional[bool] = False


@json_schema_type
class AgentTurnResponseStreamChunk(BaseModel):
    """streamed agent turn completion response."""

    event: AgentTurnResponseEvent


@json_schema_type
class AgentStepResponse(BaseModel):
    step: Step


@runtime_checkable
class Agents(Protocol):
    """Agents API for creating and interacting with agentic systems.

    Main functionalities provided by this API:
    - Create agents with specific instructions and ability to use tools.
    - Interactions with agents are grouped into sessions ("threads"), and each interaction is called a "turn".
    - Agents can be provided with various tools (see the ToolGroups and ToolRuntime APIs for more details).
    - Agents can be provided with various shields (see the Safety API for more details).
    - Agents can also use Memory to retrieve information from knowledge bases. See the RAG Tool and Vector IO APIs for more details.
    """

    @webmethod(route="/agents", method="POST", descriptive_name="create_agent")
    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse:
        """Create an agent with the given configuration.

        :param agent_config: The configuration for the agent.
        :returns: An AgentCreateResponse with the agent ID.
        """
        ...

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn", method="POST", descriptive_name="create_agent_turn"
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
        stream: Optional[bool] = False,
        documents: Optional[List[Document]] = None,
        toolgroups: Optional[List[AgentToolGroup]] = None,
        tool_config: Optional[ToolConfig] = None,
    ) -> Union[Turn, AsyncIterator[AgentTurnResponseStreamChunk]]:
        """Create a new turn for an agent.

        :param agent_id: The ID of the agent to create the turn for.
        :param session_id: The ID of the session to create the turn for.
        :param messages: List of messages to start the turn with.
        :param stream: (Optional) If True, generate an SSE event stream of the response. Defaults to False.
        :param documents: (Optional) List of documents to create the turn with.
        :param toolgroups: (Optional) List of toolgroups to create the turn with, will be used in addition to the agent's config toolgroups for the request.
        :param tool_config: (Optional) The tool configuration to create the turn with, will be used to override the agent's tool_config.
        :returns: If stream=False, returns a Turn object.
                  If stream=True, returns an SSE event stream of AgentTurnResponseStreamChunk
        """

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}/resume",
        method="POST",
        descriptive_name="resume_agent_turn",
    )
    async def resume_agent_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
        tool_responses: List[ToolResponse],
        stream: Optional[bool] = False,
    ) -> Union[Turn, AsyncIterator[AgentTurnResponseStreamChunk]]:
        """Resume an agent turn with executed tool call responses.

        When a Turn has the status `awaiting_input` due to pending input from client side tool calls, this endpoint can be used to submit the outputs from the tool calls once they are ready.

        :param agent_id: The ID of the agent to resume.
        :param session_id: The ID of the session to resume.
        :param turn_id: The ID of the turn to resume.
        :param tool_responses: The tool call responses to resume the turn with.
        :param stream: Whether to stream the response.
        :returns: A Turn object if stream is False, otherwise an AsyncIterator of AgentTurnResponseStreamChunk objects.
        """
        ...

    @webmethod(
        route="/agents/{agent_id}/session/{session_id}/turn/{turn_id}",
        method="GET",
    )
    async def get_agents_turn(
        self,
        agent_id: str,
        session_id: str,
        turn_id: str,
    ) -> Turn:
        """Retrieve an agent turn by its ID.

        :param agent_id: The ID of the agent to get the turn for.
        :param session_id: The ID of the session to get the turn for.
        :param turn_id: The ID of the turn to get.
        :returns: A Turn.
        """
        ...

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
    ) -> AgentStepResponse:
        """Retrieve an agent step by its ID.

        :param agent_id: The ID of the agent to get the step for.
        :param session_id: The ID of the session to get the step for.
        :param turn_id: The ID of the turn to get the step for.
        :param step_id: The ID of the step to get.
        :returns: An AgentStepResponse.
        """
        ...

    @webmethod(route="/agents/{agent_id}/session", method="POST", descriptive_name="create_agent_session")
    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse:
        """Create a new session for an agent.

        :param agent_id: The ID of the agent to create the session for.
        :param session_name: The name of the session to create.
        :returns: An AgentSessionCreateResponse.
        """
        ...

    @webmethod(route="/agents/{agent_id}/session/{session_id}", method="GET")
    async def get_agents_session(
        self,
        session_id: str,
        agent_id: str,
        turn_ids: Optional[List[str]] = None,
    ) -> Session:
        """Retrieve an agent session by its ID.

        :param session_id: The ID of the session to get.
        :param agent_id: The ID of the agent to get the session for.
        :param turn_ids: (Optional) List of turn IDs to filter the session by.
        """
        ...

    @webmethod(route="/agents/{agent_id}/session/{session_id}", method="DELETE")
    async def delete_agents_session(
        self,
        session_id: str,
        agent_id: str,
    ) -> None:
        """Delete an agent session by its ID.

        :param session_id: The ID of the session to delete.
        :param agent_id: The ID of the agent to delete the session for.
        """
        ...

    @webmethod(route="/agents/{agent_id}", method="DELETE")
    async def delete_agent(
        self,
        agent_id: str,
    ) -> None:
        """Delete an agent by its ID.

        :param agent_id: The ID of the agent to delete.
        """
        ...

    @webmethod(route="/agents", method="GET")
    async def list_agents(self) -> ListAgentsResponse:
        """List all agents.

        :returns: A ListAgentsResponse.
        """
        ...

    @webmethod(route="/agents/{agent_id}", method="GET")
    async def get_agent(self, agent_id: str) -> Agent:
        """Describe an agent by its ID.

        :param agent_id: ID of the agent.
        :returns: An Agent of the agent.
        """
        ...

    @webmethod(route="/agents/{agent_id}/sessions", method="GET")
    async def list_agent_sessions(
        self,
        agent_id: str,
    ) -> ListAgentSessionsResponse:
        """List all session(s) of a given agent.

        :param agent_id: The ID of the agent to list sessions for.
        :returns: A ListAgentSessionsResponse.
        """
        ...
