from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

from strong_typing.schema import json_schema_type


class ShieldType(Enum):
    """The type of safety shield."""

    llama_guard = "llama_guard"
    prompt_guard = "prompt_guard"
    code_guard = "code_guard"


@json_schema_type
@dataclass
class ShieldConfig:
    shield_type: ShieldType
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyViolation:
    violation_type: str
    details: str
    suggested_user_response: Optional[str] = None


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

# TODO(ashwin): make this better named maybe InterleavedTextMedia
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
    content: Content


# TODO: we need to document the parameters for the tool calls
class BuiltinTool(Enum):
    web_search = "web_search"
    math = "math"
    image_gen = "image_gen"
    code_interpreter = "code_interpreter"


@dataclass
class ToolDefinition:
    tool_name: Union[BuiltinTool, str]
    parameters: Optional[Dict[str, Any]] = None
    input_shields: List[ShieldConfig] = field(default_factory=list)
    output_shields: List[ShieldConfig] = field(default_factory=list)


class StopReason(Enum):
    """
    Stop reasons are used to indicate why the model stopped generating text.
    """

    not_stopped = "not_stopped"
    finished_ok = "finished_ok"
    max_tokens = "max_tokens"


@json_schema_type
@dataclass
class Message:
    role: Role

    # input to the model or output from the model
    content: Content

    # output from the model
    tool_calls: List[ToolCall] = field(default_factory=list)

    # input to the model
    tool_responses: List[ToolResponse] = field(default_factory=list)


@json_schema_type
@dataclass
class Dialog:
    message: Message
    message_history: List[Message] = None


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

class RewardModel(Enum):
    llama3_405b_reward = "llama3_405b_reward"
