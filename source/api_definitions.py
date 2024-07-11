from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union

import yaml
from agentic_system_types import (
    AgenticSystemTurn,
    ExecutionStepType,
    MemoryBank,
    MemoryBankDocument,
    SafetyViolation,
)

from finetuning_types import (
    Checkpoint,
    Dataset,
    DoraFinetuningConfig,
    FinetuningAlgorithm,
    FinetuningJobLogStream,
    FinetuningJobStatus,
    LoraFinetuningConfig,
    OptimizerConfig,
    QLoraFinetuningConfig,
    TrainingConfig,
)

from model_types import (
    BuiltinTool,
    Content,
    Dialog,
    InstructModel,
    Message,
    PretrainedModel,
    RewardModel,
    SamplingParams,
    ShieldConfig,
    StopReason,
    ToolCall,
    ToolDefinition,
    ToolResponse,
    URL,
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


@json_schema_type
@dataclass
class BatchCompletionRequest:
    content_batch: List[Content]
    model: PretrainedModel
    sampling_params: SamplingParams = SamplingParams()
    max_tokens: int = 0
    logprobs: bool = False


@json_schema_type
@dataclass
class BatchChatCompletionRequest:
    model: InstructModel
    batch_messages: List[Dialog]
    sampling_params: SamplingParams = SamplingParams()

    # zero-shot tool definitions as input to the model
    available_tools: List[Union[BuiltinTool, ToolDefinition]] = field(
        default_factory=list
    )

    max_tokens: int = 0
    logprobs: bool = False


class BatchInference(Protocol):
    """Batch inference calls"""
    def post_batch_completion(
        self,
        request: BatchCompletionRequest,
    ) -> List[CompletionResponse]: ...

    def post_batch_chat_completion(
        self,
        request: BatchChatCompletionRequest,
    ) -> List[ChatCompletionResponse]: ...


@dataclass
class AgenticSystemCreateRequest:
    uuid: str

    instructions: str
    model: InstructModel

    # zero-shot or built-in tool configurations as input to the model
    available_tools: List[ToolDefinition] = field(default_factory=list)

    # tools which aren't executable are emitted as tool calls which the users can
    # execute themselves.
    executable_tools: Set[str] = field(default_factory=set)

    memory_bank_uuids: List[str] = field(default_factory=list)

    input_shields: List[ShieldConfig] = field(default_factory=list)
    output_shields: List[ShieldConfig] = field(default_factory=list)


@json_schema_type
@dataclass
class AgenticSystemCreateResponse:
    agent_uuid: str


@json_schema_type
@dataclass
class AgenticSystemExecuteRequest:
    agent_uuid: str
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

    # TODO(ashwin): maybe add more structure here and do this as a proper tagged union
    violation: Optional[SafetyViolation] = None
    tool_call: Optional[ToolCall] = None
    tool_response_delta: Optional[ToolResponse] = None
    response_text_delta: Optional[str] = None
    retrieved_document: Optional[MemoryBankDocument] = None

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


class MemoryBanks(Protocol):
    @webmethod(route="/memory_banks/create")
    def create_memory_bank(
        self,
        bank_uuid: str,
        bank_name: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_banks/get")
    def get_memory_banks(
        self,
    ) -> List[MemoryBank]: ...

    @webmethod(route="/memory_banks/insert")
    def post_insert_memory_documents(
        self,
        bank_uuid: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_banks/delete")
    def post_delete_memory_documents(
        self,
        bank_uuid: str,
        document_uuids: List[str],
    ) -> None: ...

    @webmethod(route="/memory_banks/drop")
    def remove_memory_bank(
        self,
        bank_uuid: str,
    ) -> None: ...


@dataclass
class KPromptGenerations:
    prompt: Message
    message_history: List[Message]
    k_generations: List[Message]


@json_schema_type
@dataclass
class MessageScore:
    """A single message and its score."""

    message: Message
    score: float


@json_schema_type
@dataclass
class KScoredPromptGenerations:
    prompt: Message
    k_scored_generations: List[MessageScore]


@json_schema_type
@dataclass
class RewardScoringRequest:
    """Request to score a reward function. A list of prompts and a list of responses per prompt."""

    prompt_generations: List[KPromptGenerations]
    model: RewardModel


@json_schema_type
@dataclass
class RewardScoringResponse:
    """Response from the reward scoring. Batch of (prompt, response, score) tuples that pass the threshold."""

    scored_generations: List[KScoredPromptGenerations]


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

    prompts: List[Message]
    filtering_function: FilteringFunction = FilteringFunction.none
    reward_scoring: Optional[RewardScoring] = None


@json_schema_type
@dataclass
class SyntheticDataGenerationResponse:
    """Response from the synthetic data generation. Batch of (prompt, response, score) tuples that pass the threshold."""

    synthetic_data: List[KScoredPromptGenerations]
    statistics: Optional[Dict[str, Any]] = None


class SyntheticDataGeneration(Protocol):
    @webmethod(route="/synthetic_data_generation/generate")
    def post_generate(
        self,
        request: SyntheticDataGenerationRequest,
    ) -> Union[SyntheticDataGenerationResponse]: ...


@json_schema_type
@dataclass
class CreateDatasetRequest:
    """Request to create a dataset."""

    uuid: str
    dataset: Dataset


class Datasets(Protocol):
    @webmethod(route="/datasets/create")
    def create_dataset(
        self,
        request: CreateDatasetRequest,
    ) -> None: ...

    @webmethod(route="/datasets/get")
    def get_dataset(
        self,
        dataset_id: str,
    ) -> Dataset: ...

    @webmethod(route="/datasets/delete")
    def delete_dataset(
        self,
        dataset_id: str,
    ) -> None: ...


@json_schema_type
@dataclass
class FinetuningTrainRequest:
    """Request to finetune a model."""

    job_uuid: str

    model: PretrainedModel
    dataset: Dataset
    validation_dataset: Dataset

    algorithm: FinetuningAlgorithm
    algorithm_config: Union[
        LoraFinetuningConfig, QLoraFinetuningConfig, DoraFinetuningConfig
    ]

    optimizer_config: OptimizerConfig
    training_config: TrainingConfig

    # TODO: define these
    hyperparam_search_config: Dict[str, Any]
    logger_config: Dict[str, Any]


@json_schema_type
@dataclass
class FinetuningJobStatusResponse:
    """Status of a finetuning job."""

    job_uuid: str
    status: FinetuningJobStatus

    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    resources_allocated: Optional[Dict[str, Any]] = None

    checkpoints: List[Checkpoint] = field(default_factory=list)


@json_schema_type
@dataclass
class FinetuningJobArtifactsResponse:
    """Artifacts of a finetuning job."""

    job_uuid: str
    checkpoints: List[Checkpoint] = field(default_factory=list)

    # TODO(ashwin): metrics, evals


class Finetuning(Protocol):
    @webmethod(route="/finetuning/text_generation/train")
    def post_train(
        self,
        request: FinetuningTrainRequest,
    ) -> None: ...

    # sends SSE stream of logs
    @webmethod(route="/finetuning/job/logs")
    def get_training_log_stream(self, job_uuid: str) -> FinetuningJobLogStream: ...

    @webmethod(route="/finetuning/job/status")
    def get_training_job_status(self, job_uuid: str) -> FinetuningJobStatusResponse: ...

    @webmethod(route="/finetuning/job/cancel")
    def cancel_training_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/finetuning/job/artifacts")
    def get_training_job_artifacts(
        self, job_uuid: str
    ) -> FinetuningJobArtifactsResponse: ...


class LlamaStackEndpoints(
    Inference,
    AgenticSystem,
    RewardScoring,
    SyntheticDataGeneration,
    Datasets,
    Finetuning,
    MemoryBanks,
): ...


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
