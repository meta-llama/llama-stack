from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from model_types import (
    BuiltinTool,
    Content,
    InstructModel,
    Message,
    PretrainedModel,
    SamplingParams,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolResponse,
)

from strong_typing.schema import json_schema_type


class ExecutionStepType(Enum):
    """The type of execution step."""

    model_inference = "model_inference"
    tool_execution = "tool_execution"
    safety_filtering = "safety_filtering"
    memory_retrieval = "memory_retrieval"


@dataclass
class ExecutionStepBase:
    """An agentic system turn can consist of one or more such execution steps."""

    step_type: ExecutionStepType
    uuid: str


@dataclass
class ModelInferenceStep(ExecutionStepBase):
    step_type = ExecutionStepType.model_inference
    text: str
    logprobs: Optional[Dict[str, Any]] = None


@dataclass
class ToolExecutionStep(ExecutionStepBase):
    step_type = ExecutionStepType.tool_execution

    # we could be calling multiple tools in a single step (in parallel)
    tool_calls: List[ToolCall]
    tool_responses: List[ToolResponse]


@dataclass
class SafetyViolation:
    violation_type: str
    details: str
    suggested_user_response: Optional[str] = None


@dataclass
class SafetyFilteringStep(ExecutionStepBase):
    step_type = ExecutionStepType.safety_filtering
    violation: Optional[SafetyViolation] = None


@dataclass
class IndexedMemoryDocument:
    index_id: str
    content: str


@dataclass
class MemoryRetrievalStep(ExecutionStepBase):
    step_type = ExecutionStepType.memory_retrieval
    documents: List[IndexedMemoryDocument]
    scores: List[float]


ExecutionStep = Union[
    ModelInferenceStep,
    ToolExecutionStep,
    SafetyFilteringStep,
    MemoryRetrievalStep,
]


@json_schema_type
@dataclass
class AgenticSystemTurn:
    """A single turn in an interaction with an Agentic System."""

    user_messages: List[Message]
    steps: List[ExecutionStep]
    response_message: Message
