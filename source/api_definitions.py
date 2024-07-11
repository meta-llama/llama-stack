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

from post_training_types import (
    Checkpoint,
    Dataset,
    DoraFinetuningConfig,
    DPOAlignmentConfig,
    FinetuningAlgorithm,
    LoraFinetuningConfig,
    OptimizerConfig,
    PostTrainingJobLogStream,
    PostTrainingJobStatus,
    QLoraFinetuningConfig,
    RLHFAlgorithm,
    TrainingConfig,
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
    model: InstructModel
    dialog: Dialog
    sampling_params: SamplingParams = SamplingParams()

    # zero-shot tool definitions as input to the model
    available_tools: List[ToolDefinition] = field(default_factory=list)

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


@json_schema_type
@dataclass
class BatchCompletionRequest:
    model: PretrainedModel
    content_batch: List[Content]
    sampling_params: SamplingParams = SamplingParams()
    max_tokens: int = 0
    logprobs: bool = False


@json_schema_type
@dataclass
class BatchChatCompletionRequest:
    model: InstructModel
    batch_dialogs: List[Dialog]
    sampling_params: SamplingParams = SamplingParams()

    # zero-shot tool definitions as input to the model
    available_tools: List[ToolDefinition] = field(default_factory=list)

    max_tokens: int = 0
    logprobs: bool = False


class Inference(Protocol):

    @webmethod(route="/inference/completion")
    def post_completion(
        self,
        request: CompletionRequest,
    ) -> Union[CompletionResponse, CompletionResponseStreamChunk]: ...

    @webmethod(route="/inference/chat_completion")
    def post_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, ChatCompletionResponseStreamChunk]: ...

    @webmethod(route="/inference/batch_completion")
    def post_batch_completion(
        self,
        request: BatchCompletionRequest,
    ) -> List[CompletionResponse]: ...

    @webmethod(route="/inference/batch_chat_completion")
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
    def post_create_memory_bank(
        self,
        bank_uuid: str,
        bank_name: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_banks/get")
    def get_memory_banks(
        self
    ) -> List[MemoryBank]: ...

    @webmethod(route="/memory_banks/drop")
    def delete_memory_bank(
        self,
        bank_uuid: str,
    ) -> str: ...

    @webmethod(route="/memory_bank/insert")
    def post_insert_memory_documents(
        self,
        bank_uuid: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_bank/update")
    def post_update_memory_documents(
        self,
        bank_uuid: str,
        documents: List[MemoryBankDocument],
    ) -> None: ...

    @webmethod(route="/memory_bank/get")
    def get_memory_documents(
        self,
        bank_uuid: str,
        document_uuids: List[str],
    ) -> List[MemoryBankDocument]: ...

    @webmethod(route="/memory_bank/delete")
    def delete_memory_documents(
        self,
        bank_uuid: str,
        document_uuids: List[str],
    ) -> List[str]: ...


@dataclass
class KPromptGenerations:
    dialog: Dialog
    k_generations: List[Message]


@json_schema_type
@dataclass
class ScoredMessage:
    message: Message
    score: float


@json_schema_type
@dataclass
class KScoredPromptGenerations:
    prompt: Message
    k_scored_generations: List[ScoredMessage]


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
class PostTrainingSFTRequest:
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
class PostTrainingRLHFRequest:
    """Request to finetune a model."""

    job_uuid: str

    finetuned_model: URL

    dataset: Dataset
    validation_dataset: Dataset

    algorithm: RLHFAlgorithm
    algorithm_config: Union[DPOAlignmentConfig]

    optimizer_config: OptimizerConfig
    training_config: TrainingConfig

    # TODO: define these
    hyperparam_search_config: Dict[str, Any]
    logger_config: Dict[str, Any]


@json_schema_type
@dataclass
class PostTrainingJobStatusResponse:
    """Status of a finetuning job."""

    job_uuid: str
    status: PostTrainingJobStatus

    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    resources_allocated: Optional[Dict[str, Any]] = None

    checkpoints: List[Checkpoint] = field(default_factory=list)


@json_schema_type
@dataclass
class PostTrainingJobArtifactsResponse:
    """Artifacts of a finetuning job."""

    job_uuid: str
    checkpoints: List[Checkpoint] = field(default_factory=list)

    # TODO(ashwin): metrics, evals


class PostTraining(Protocol):
    @webmethod(route="/post_training/supervised_fine_tune/")
    def post_supervised_fine_tune(
        self,
        request: PostTrainingSFTRequest,
    ) -> None: ...

    @webmethod(route="/post_training/preference_optimize/")
    def post_preference_optimize(
        self,
        request: PostTrainingRLHFRequest,
    ) -> None: ...

    # sends SSE stream of logs
    @webmethod(route="/post_training/job/logs")
    def get_training_log_stream(self, job_uuid: str) -> PostTrainingJobLogStream: ...

    @webmethod(route="/post_training/job/status")
    def get_training_job_status(
        self, job_uuid: str
    ) -> PostTrainingJobStatusResponse: ...

    @webmethod(route="/post_training/job/cancel")
    def cancel_training_job(self, job_uuid: str) -> None: ...

    @webmethod(route="/post_training/job/artifacts")
    def get_training_job_artifacts(
        self, job_uuid: str
    ) -> PostTrainingJobArtifactsResponse: ...


class LlamaStackEndpoints(
    Inference,
    AgenticSystem,
    RewardScoring,
    SyntheticDataGeneration,
    Datasets,
    PostTraining,
    MemoryBanks,
): ...


if __name__ == "__main__":
    print("Converting the spec to YAML (openapi.yaml) and HTML (openapi.html)")
    spec = Specification(
        LlamaStackEndpoints,
        Options(
            server=Server(url="http://any-hosted-llama-stack.com"),
            info=Info(
                title="[DRAFT] Llama Stack Specification",
                version="0.0.1",
                description="""Meta has built out a fairly sophisticated platform internally to post train, evaluate, and 
                serve Llama models to support Meta’s products. Given the newer capabilities of the llama models, 
                the model development and model serving capabilities of the platform need to be enhanced in 
                specific ways in order to best leverage the models. For example, the inference platform needs 
                to support code execution to take advantage of the built-in knowledge of tools of the model. 
                The largest models are of high enough quality to be used to generate synthetic data or be used 
                as reward models. There are specific fine tuning and quantization techniques that we have found 
                result in the best performing Llama models. We would like to share ways in which an LLM Ops 
                toolchain can be designed by leveraging our learnings in getting Llama models to power Meta’s products.

                In addition, the Llama 3 models Meta will release in July should not just be seen as a model, but 
                really as a system starting the transition towards an entity capable of performing "agentic" tasks 
                which require the ability to act as the central planner and break a task down and perform multi-step 
                reasoning and call tools for specific operations. In addition, there needs to be general model-level 
                safety checks as well as task-specific safety checks that are performed at a system level. 

                We are defining the Llama Stack as a set of APIs and standards by synthesizing our learnings while 
                working with Llama models. The APIs are divided into the llama-toolchain-api and the llama-agentic-system-api. 
                These APIs provide a coherent way for model developers to fine tune and serve Llama models, and agentic app 
                developers to leverage all the capabilities of the Llama models seamlessly. We would like to work with the 
                ecosystem to enhance and simplify the API. In addition, we will be releasing a plug-in architecture to allow 
                creating distributions of the llama stack with different implementations.


                This is the specification of the llama stack that provides 
                a set of endpoints and their corresponding interfaces that are tailored to 
                best leverage Llama Models. The specification is still in draft and subject to change.""",
            ),
        ),
    )
    with open("openapi.yaml", "w", encoding="utf-8") as fp:
        yaml.dump(spec.get_json(), fp, allow_unicode=True)

    with open("openapi.html", "w") as fp:
        spec.write_html(fp, pretty_print=True)
