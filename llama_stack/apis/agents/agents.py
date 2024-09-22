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
from llama_stack.apis.common.deployment_types import *  # noqa: F403
from llama_stack.apis.inference import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from llama_stack.apis.memory import *  # noqa: F403


@json_schema_type
class Attachment(BaseModel):
    content: InterleavedTextMedia | URL
    mime_type: str


class AgentTool(Enum):
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"

    function_call = "function_call"
    memory = "memory"


class ToolDefinitionCommon(BaseModel):
    input_shields: Optional[List[str]] = Field(default_factory=list)
    output_shields: Optional[List[str]] = Field(default_factory=list)


class SearchEngineType(Enum):
    bing = "bing"
    brave = "brave"


@json_schema_type
class SearchToolDefinition(ToolDefinitionCommon):
    # NOTE: brave_search is just a placeholder since model always uses
    # brave_search as tool call name
    type: Literal[AgentTool.brave_search.value] = AgentTool.brave_search.value
    api_key: str
    engine: SearchEngineType = SearchEngineType.brave
    remote_execution: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class WolframAlphaToolDefinition(ToolDefinitionCommon):
    type: Literal[AgentTool.wolfram_alpha.value] = AgentTool.wolfram_alpha.value
    api_key: str
    remote_execution: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class PhotogenToolDefinition(ToolDefinitionCommon):
    type: Literal[AgentTool.photogen.value] = AgentTool.photogen.value
    remote_execution: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class CodeInterpreterToolDefinition(ToolDefinitionCommon):
    type: Literal[AgentTool.code_interpreter.value] = AgentTool.code_interpreter.value
    enable_inline_code_execution: bool = True
    remote_execution: Optional[RestAPIExecutionConfig] = None


@json_schema_type
class FunctionCallToolDefinition(ToolDefinitionCommon):
    type: Literal[AgentTool.function_call.value] = AgentTool.function_call.value
    function_name: str
    description: str
    parameters: Dict[str, ToolParamDefinition]
    remote_execution: Optional[RestAPIExecutionConfig] = None


class _MemoryBankConfigCommon(BaseModel):
    bank_id: str


class AgentVectorMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal[MemoryBankType.vector.value] = MemoryBankType.vector.value


class AgentKeyValueMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal[MemoryBankType.keyvalue.value] = MemoryBankType.keyvalue.value
    keys: List[str]  # what keys to focus on


class AgentKeywordMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal[MemoryBankType.keyword.value] = MemoryBankType.keyword.value


class AgentGraphMemoryBankConfig(_MemoryBankConfigCommon):
    type: Literal[MemoryBankType.graph.value] = MemoryBankType.graph.value
    entities: List[str]  # what entities to focus on


MemoryBankConfig = Annotated[
    Union[
        AgentVectorMemoryBankConfig,
        AgentKeyValueMemoryBankConfig,
        AgentKeywordMemoryBankConfig,
        AgentGraphMemoryBankConfig,
    ],
    Field(discriminator="type"),
]


class MemoryQueryGenerator(Enum):
    default = "default"
    llm = "llm"
    custom = "custom"


class DefaultMemoryQueryGeneratorConfig(BaseModel):
    type: Literal[MemoryQueryGenerator.default.value] = (
        MemoryQueryGenerator.default.value
    )
    sep: str = " "


class LLMMemoryQueryGeneratorConfig(BaseModel):
    type: Literal[MemoryQueryGenerator.llm.value] = MemoryQueryGenerator.llm.value
    model: str
    template: str


class CustomMemoryQueryGeneratorConfig(BaseModel):
    type: Literal[MemoryQueryGenerator.custom.value] = MemoryQueryGenerator.custom.value


MemoryQueryGeneratorConfig = Annotated[
    Union[
        DefaultMemoryQueryGeneratorConfig,
        LLMMemoryQueryGeneratorConfig,
        CustomMemoryQueryGeneratorConfig,
    ],
    Field(discriminator="type"),
]


@json_schema_type
class MemoryToolDefinition(ToolDefinitionCommon):
    type: Literal[AgentTool.memory.value] = AgentTool.memory.value
    memory_bank_configs: List[MemoryBankConfig] = Field(default_factory=list)
    # This config defines how a query is generated using the messages
    # for memory bank retrieval.
    query_generator_config: MemoryQueryGeneratorConfig = Field(
        default=DefaultMemoryQueryGeneratorConfig()
    )
    max_tokens_in_context: int = 4096
    max_chunks: int = 10


AgentToolDefinition = Annotated[
    Union[
        SearchToolDefinition,
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
    violation: Optional[SafetyViolation]


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

    input_shields: Optional[List[str]] = Field(default_factory=list)
    output_shields: Optional[List[str]] = Field(default_factory=list)

    tools: Optional[List[AgentToolDefinition]] = Field(default_factory=list)
    tool_choice: Optional[ToolChoice] = Field(default=ToolChoice.auto)
    tool_prompt_format: Optional[ToolPromptFormat] = Field(
        default=ToolPromptFormat.json
    )

    max_infer_iters: int = 10


@json_schema_type
class AgentConfig(AgentConfigCommon):
    model: str
    instructions: str
    enable_session_persistence: bool


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
    event_type: Literal[AgentTurnResponseEventType.step_start.value] = (
        AgentTurnResponseEventType.step_start.value
    )
    step_type: StepType
    step_id: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


@json_schema_type
class AgentTurnResponseStepCompletePayload(BaseModel):
    event_type: Literal[AgentTurnResponseEventType.step_complete.value] = (
        AgentTurnResponseEventType.step_complete.value
    )
    step_type: StepType
    step_details: Step


@json_schema_type
class AgentTurnResponseStepProgressPayload(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    event_type: Literal[AgentTurnResponseEventType.step_progress.value] = (
        AgentTurnResponseEventType.step_progress.value
    )
    step_type: StepType
    step_id: str

    model_response_text_delta: Optional[str] = None
    tool_call_delta: Optional[ToolCallDelta] = None
    tool_response_text_delta: Optional[str] = None


@json_schema_type
class AgentTurnResponseTurnStartPayload(BaseModel):
    event_type: Literal[AgentTurnResponseEventType.turn_start.value] = (
        AgentTurnResponseEventType.turn_start.value
    )
    turn_id: str


@json_schema_type
class AgentTurnResponseTurnCompletePayload(BaseModel):
    event_type: Literal[AgentTurnResponseEventType.turn_complete.value] = (
        AgentTurnResponseEventType.turn_complete.value
    )
    turn: Turn


@json_schema_type
class AgentTurnResponseEvent(BaseModel):
    """Streamed agent execution response."""

    payload: Annotated[
        Union[
            AgentTurnResponseStepStartPayload,
            AgentTurnResponseStepProgressPayload,
            AgentTurnResponseStepCompletePayload,
            AgentTurnResponseTurnStartPayload,
            AgentTurnResponseTurnCompletePayload,
        ],
        Field(discriminator="event_type"),
    ]


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
    attachments: Optional[List[Attachment]] = None

    stream: Optional[bool] = False


@json_schema_type
class AgentTurnResponseStreamChunk(BaseModel):
    event: AgentTurnResponseEvent


@json_schema_type
class AgentStepResponse(BaseModel):
    step: Step


class Agents(Protocol):
    @webmethod(route="/agents/create")
    async def create_agent(
        self,
        agent_config: AgentConfig,
    ) -> AgentCreateResponse: ...

    @webmethod(route="/agents/turn/create")
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
    ) -> AgentTurnResponseStreamChunk: ...

    @webmethod(route="/agents/turn/get")
    async def get_agents_turn(
        self,
        agent_id: str,
        turn_id: str,
    ) -> Turn: ...

    @webmethod(route="/agents/step/get")
    async def get_agents_step(
        self, agent_id: str, turn_id: str, step_id: str
    ) -> AgentStepResponse: ...

    @webmethod(route="/agents/session/create")
    async def create_agent_session(
        self,
        agent_id: str,
        session_name: str,
    ) -> AgentSessionCreateResponse: ...

    @webmethod(route="/agents/session/get")
    async def get_agents_session(
        self,
        agent_id: str,
        session_id: str,
        turn_ids: Optional[List[str]] = None,
    ) -> Session: ...

    @webmethod(route="/agents/session/delete")
    async def delete_agents_session(self, agent_id: str, session_id: str) -> None: ...

    @webmethod(route="/agents/delete")
    async def delete_agents(
        self,
        agent_id: str,
    ) -> None: ...
