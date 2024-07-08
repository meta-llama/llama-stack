from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

import yaml

from pyopenapi import Info, Options, Server, Specification, webmethod
from strong_typing.schema import json_schema_type


@json_schema_type(
    schema={"type": "string", "format": "uri", "pattern": "^(https?://|file://|data:)"}
)
@dataclass
class URL:
    url: str

    def __str__(self) -> str:
        return self.url


@json_schema_type
@dataclass
class Attachment:
    """
    Attachments are used to refer to external resources, such as images, videos, audio, etc.

    """

    url: URL
    mime_type: str


Content = Union[
    str,
    Attachment,
    List[Union[str, Attachment]],
]


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class StopReason(Enum):
    """
    Stop reasons are used to indicate why the model stopped generating text.
    """

    not_stopped = "not_stopped"
    finished_ok = "finished_ok"
    max_tokens = "max_tokens"


@dataclass
class ToolCall:
    """
    A tool call is a request to a tool.
    """

    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolResponse:
    tool_name: str
    response: str


@dataclass
class ToolDefinition:
    tool_name: str
    parameters: Dict[str, Any]


@json_schema_type
@dataclass
class Message:
    role: Role

    # input to the model or output from the model
    content: Content

    # zero-shot tool definitions as input to the model
    tool_definitions: List[ToolDefinition] = field(default_factory=list)

    # output from the model
    tool_calls: List[ToolCall] = field(default_factory=list)

    # input to the model
    tool_responses: List[ToolResponse] = field(default_factory=list)


@json_schema_type
@dataclass
class CompletionResponse:
    """Normal completion response."""
    content: Content
    stop_reason: StopReason
    logprobs: Optional[Dict[str, Any]] = None


@json_schema_type
@dataclass
class StreamedCompletionResponse:
    """streamed completion response."""
    text_delta: str
    stop_reason: StopReason
    logprobs: Optional[Dict[str, Any]] = None


@json_schema_type
@dataclass
class ChatCompletionResponse:
    """Normal chat completion response."""

    content: Content
    stop_reason: StopReason
    tool_calls: List[ToolCall] = field(default_factory=list)
    logprobs: Optional[Dict[str, Any]] = None


@json_schema_type
@dataclass
class StreamedChatCompletionResponse:
    """Streamed chat completion response."""

    text_delta: str
    stop_reason: StopReason
    tool_call: Optional[ToolCall] = None


@dataclass
class SamplingParams:
    temperature: float = 0.0
    strategy: str = "greedy"
    top_p: float = 0.95
    top_k: int = 0


class PretrainedModel(Enum):
    llama3_8b = "llama3_8b"
    llama3_70b = "llama3_70b"


class InstructModel(Enum):
    llama3_8b_chat = "llama3_8b_chat"
    llama3_70b_chat = "llama3_70b_chat"


@json_schema_type
@dataclass
class CompletionRequest:
    content: Content
    model: PretrainedModel = PretrainedModel.llama3_8b
    sampling_params: SamplingParams = SamplingParams()
    max_tokens: int = 0
    stream: bool = False
    logprobs: bool = False


@json_schema_type
@dataclass
class ChatCompletionRequest:
    message: Message
    message_history: List[Message] = None
    model: InstructModel = InstructModel.llama3_8b_chat
    sampling_params: SamplingParams = SamplingParams()
    max_tokens: int = 0
    stream: bool = False
    logprobs: bool = False


class Inference(Protocol):

    def post_completion(
        self,
        request: CompletionRequest,
    ) -> Union[CompletionResponse, StreamedCompletionResponse]: ...

    def post_chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> Union[ChatCompletionResponse, StreamedChatCompletionResponse]: ...



@json_schema_type
@dataclass
class AgenticSystemExecuteRequest:
    message: Message
    message_history: List[Message] = None
    model: InstructModel = InstructModel.llama3_8b_chat
    sampling_params: SamplingParams = SamplingParams()

class AgenticSystem(Protocol):

    @webmethod(route="/agentic/system/execute")
    def create_agentic_system_execute(self,) -> str: ...


class Endpoint(Inference, AgenticSystem): ...


if __name__ == "__main__":
    print("Converting the spec to YAML (openapi.yaml) and HTML (openapi.html)")
    spec = Specification(
        Endpoint,
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
