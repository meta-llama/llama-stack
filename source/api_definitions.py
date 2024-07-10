from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Union

import yaml
from agentic_system_types import (
    AgenticSystemTurn,
    ExecutionStepType,
    IndexedMemoryDocument,
    SafetyViolation,
)

from model_types import (
    BuiltinTool,
    Content,
    InstructModel,
    Message,
    PretrainedModel,
    SamplingParams,
    StopReason,
    ShieldConfig,
    ToolCall,
    ToolDefinition,
    ToolResponse,
)

from pyopenapi import Info, Options, Server, Specification, webmethod
from strong_typing.schema import json_schema_type


@json_schema_type
@dataclass
class CompletionRequest:
    content: Content
    model: PretrainedModel
    sampling_params: SamplingParams = SamplingParams()
    max_tokens: int = 0
    stream: bool = False
    logprobs: bool = False


@json_schema_type
@dataclass
class CompletionResponse:
    """Normal completion response."""

    content: Content
    stop_reason: Optional[StopReason] = None
    logprobs: Optional[Dict[str, Any]] = None


@json_schema_type
@dataclass
class CompletionResponseStreamChunk:
    """streamed completion response."""

    text_delta: str
    stop_reason: Optional[StopReason] = None
    logprobs: Optional[Dict[str, Any]] = None


@json_schema_type
@dataclass
class ChatCompletionRequest:
    message: Message
    model: InstructModel
    message_history: List[Message] = None
    sampling_params: SamplingParams = SamplingParams()

    # zero-shot tool definitions as input to the model
    available_tools: List[Union[BuiltinTool, ToolDefinition]] = field(
        default_factory=list
    )

    max_tokens: int = 0
    stream: bool = False
    logprobs: bool = False


@json_schema_type
@dataclass
class ChatCompletionResponse:
    """Normal chat completion response."""

    content: Content

    # note: multiple tool calls can be generated in a single response
    tool_calls: List[ToolCall] = field(default_factory=list)

    stop_reason: Optional[StopReason] = None
    logprobs: Optional[Dict[str, Any]] = None


@json_schema_type
@dataclass
class ChatCompletionResponseStreamChunk:
    """Streamed chat completion response. The actual response is a series of such objects."""

    text_delta: str
    stop_reason: Optional[StopReason] = None
    tool_call: Optional[ToolCall] = None


class Inference(Protocol):

    def post_completion(
        self,
        request: CompletionRequest,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]: ...

    def post_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]: ...


@dataclass
class AgenticSystemCreateRequest:
    instructions: str
    model: InstructModel

    # zero-shot or built-in tool configurations as input to the model
    available_tools: List[ToolDefinition] = field(default_factory=list)

    # tools which aren't executable are emitted as tool calls which the users can
    # execute themselves.
    executable_tools: Set[str] = field(default_factory=set)

    input_shields: List[ShieldConfig] = field(default_factory=list)
    output_shields: List[ShieldConfig] = field(default_factory=list)


@json_schema_type
@dataclass
class AgenticSystemCreateResponse:
    agent_id: str


@json_schema_type
@dataclass
class AgenticSystemExecuteRequest:
    agent_id: str
    messages: List[Message]
    turn_history: List[AgenticSystemTurn] = None
    stream: bool = False


@json_schema_type
@dataclass
class AgenticSystemExecuteResponse:
    """non-stream response from the agentic system."""

    turn: AgenticSystemTurn


class AgenticSystemExecuteResponseEventType(Enum):
    """The type of event."""

    step_start = "step_start"
    step_end = "step_end"
    step_progress = "step_progress"


@json_schema_type
@dataclass
class AgenticSystemExecuteResponseStreamChunk:
    """Streamed agent execution response."""

    event_type: AgenticSystemExecuteResponseEventType

    step_uuid: str
    step_type: ExecutionStepType

    violation: Optional[SafetyViolation] = None
    tool_call: Optional[ToolCall] = None
    tool_response_delta: Optional[ToolResponse] = None
    response_text_delta: Optional[str] = None
    retrieved_document: Optional[IndexedMemoryDocument] = None

    stop_reason: Optional[StopReason] = None


class AgenticSystem(Protocol):

    @webmethod(route="/agentic_system/create")
    def create_agentic_system(
        self,
        request: AgenticSystemCreateRequest,
    ) -> AgenticSystemCreateResponse: ...

    @webmethod(route="/agentic_system/execute")
    def create_agentic_system_execute(
        self,
        request: AgenticSystemExecuteRequest,
    ) -> Union[
        AgenticSystemExecuteResponse, AgenticSystemExecuteResponseStreamChunk
    ]: ...

    @webmethod(route="/agentic_system/delete")
    def delete_agentic_system(
        self,
        agent_id: str,
    ) -> None: ...


@json_schema_type
@dataclass
class RewardScoringRequest:
    """Request to score a reward function. A list of prompts and a list of responses per prompt."""

    prompt_responses: List[Tuple(prompt:str, List[response:str])]

@json_schema_type
@dataclass
class RewardScoringResponse:
    """Response from the reward scoring. Batch of (prompt, response, score) tuples that pass the threshold."""

    prompt_responses: List[Tuple(prompt:str, List[Tuple(response:str, score:float)])]

class RewardScoring(Protocol):
    @webmethod(route="/reward_scoring/score")
    def post_score(
        self,
        request: RewardScoringRequest,
    ) -> Union[RewardScoringResponse]: ...

class FilteringFunction(Enum):
    """The type of filtering function."""

    none = "none"
    random = "random"
    top_k = "top_k"
    top_p = "top_p"
    top_k_top_p = "top_k_top_p"
    sigmoid = "sigmoid"

@json_schema_type
@dataclass
class SyntheticDataGenerationRequest:
    """Request to generate synthetic data. A small batch of prompts and a filtering function"""

    prompts: List[str]
    filtering_function: FilteringFunction = FilteringFunction.none
    reward_scoring: RewardScoring

@json_schema_type
@dataclass
class SyntheticDataGenerationResponse:
    """Response from the synthetic data generation. Batch of (prompt, response, score) tuples that pass the threshold."""

    synthetic_data: List[Tuple(prompt:str, response:str, score:float)]
    """The actual synthetic data"""
    statistics: Dict[str, float]
    """Statistics on how many prompts were generated and how many were filtered out"""

class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic_data_generation/generate")
    def post_generate(
        self,
        request: SyntheticDataGenerationRequest,
    ) -> Union[SyntheticDataGenerationResponse]: ...



class LlamaStackEndpoints(Inference, AgenticSystem, RewardScoring, SyntheticDataGeneration): ...


if __name__ == "__main__":
    print("Converting the spec to YAML (openapi.yaml) and HTML (openapi.html)")
    spec = Specification(
        LlamaStackEndpoints,
        Options(
            server=Server(url="http://llama.meta.com"),
            info=Info(
                title="Llama Stack specification",
                version="0.1",
                description="This is the llama stack",
            ),
        ),
    )
    with open("openapi.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(spec.get_json(), fp, allow_unicode=True)

    with open("openapi.html", "w") as fp:
        spec.write_html(fp, pretty_print=True)
