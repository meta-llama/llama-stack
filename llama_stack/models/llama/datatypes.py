# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

import base64
from enum import Enum
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from typing_extensions import Annotated

from llama_stack.schema_utils import json_schema_type, register_schema

# The goal is that these set of types are relevant for all Llama models.
# That isn't the current state yet -- e.g., BuiltinTool is somewhat specific to
# the llama3 series of models.


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"


class BuiltinTool(Enum):
    brave_search = "brave_search"
    wolfram_alpha = "wolfram_alpha"
    photogen = "photogen"
    code_interpreter = "code_interpreter"


Primitive = Union[str, int, float, bool, None]
RecursiveType = Union[Primitive, List[Primitive], Dict[str, Primitive]]


class ToolCall(BaseModel):
    call_id: str
    tool_name: Union[BuiltinTool, str]
    # Plan is to deprecate the Dict in favor of a JSON string
    # that is parsed on the client side instead of trying to manage
    # the recursive type here.
    # Making this a union so that client side can start prepping for this change.
    # Eventually, we will remove both the Dict and arguments_json field,
    # and arguments will just be a str
    arguments: Union[str, Dict[str, RecursiveType]]
    arguments_json: Optional[str] = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


class ToolPromptFormat(Enum):
    """Prompt format for calling custom / zero shot tools.

    :cvar json: JSON format for calling tools. It takes the form:
        {
            "type": "function",
            "function" : {
                "name": "function_name",
                "description": "function_description",
                "parameters": {...}
            }
        }
    :cvar function_tag: Function tag format, pseudo-XML. This looks like:
        <function=function_name>(parameters)</function>

    :cvar python_list: Python list. The output is a valid Python expression that can be
        evaluated to a list. Each element in the list is a function call. Example:
        ["function_name(param1, param2)", "function_name(param1, param2)"]
    """

    json = "json"
    function_tag = "function_tag"
    python_list = "python_list"


class StopReason(Enum):
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


class RawMediaItem(BaseModel):
    type: Literal["image"] = "image"
    data: bytes | BytesIO

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("data")
    def serialize_data(self, data: Optional[bytes], _info):
        if data is None:
            return None
        return base64.b64encode(data).decode("utf-8")

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v):
        if isinstance(v, str):
            return base64.b64decode(v)
        return v


class RawTextItem(BaseModel):
    type: Literal["text"] = "text"
    text: str


RawContentItem = Annotated[Union[RawTextItem, RawMediaItem], Field(discriminator="type")]

RawContent = str | RawContentItem | List[RawContentItem]


class RawMessage(BaseModel):
    role: Literal["user"] | Literal["system"] | Literal["tool"] | Literal["assistant"]
    content: RawContent

    # This is for RAG but likely should be absorbed into content
    context: Optional[RawContent] = None

    # These are for the output message coming from the assistant
    stop_reason: Optional[StopReason] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


register_schema(ToolCall)


@json_schema_type
class ToolParamDefinition(BaseModel):
    param_type: str
    description: Optional[str] = None
    required: Optional[bool] = True
    default: Optional[Any] = None


@json_schema_type
class ToolDefinition(BaseModel):
    tool_name: Union[BuiltinTool, str]
    description: Optional[str] = None
    parameters: Optional[Dict[str, ToolParamDefinition]] = None

    @field_validator("tool_name", mode="before")
    @classmethod
    def validate_field(cls, v):
        if isinstance(v, str):
            try:
                return BuiltinTool(v)
            except ValueError:
                return v
        return v


@json_schema_type
class GreedySamplingStrategy(BaseModel):
    type: Literal["greedy"] = "greedy"


@json_schema_type
class TopPSamplingStrategy(BaseModel):
    type: Literal["top_p"] = "top_p"
    temperature: Optional[float] = Field(..., gt=0.0)
    top_p: Optional[float] = 0.95


@json_schema_type
class TopKSamplingStrategy(BaseModel):
    type: Literal["top_k"] = "top_k"
    top_k: int = Field(..., ge=1)


SamplingStrategy = Annotated[
    Union[GreedySamplingStrategy, TopPSamplingStrategy, TopKSamplingStrategy],
    Field(discriminator="type"),
]
register_schema(SamplingStrategy, name="SamplingStrategy")


@json_schema_type
class SamplingParams(BaseModel):
    """Sampling parameters.

    :param strategy: The sampling strategy.
    :param max_tokens: The maximum number of tokens that can be generated in the completion. The token count of
        your prompt plus max_tokens cannot exceed the model's context length.
    :param repetition_penalty: Number between -2.0 and 2.0. Positive values penalize new tokens
        based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    :param stop: Up to 4 sequences where the API will stop generating further tokens.
        The returned text will not contain the stop sequence.
    """

    strategy: SamplingStrategy = Field(default_factory=GreedySamplingStrategy)

    max_tokens: Optional[int] = 0
    repetition_penalty: Optional[float] = 1.0
    stop: Optional[List[str]] = None


class CheckpointQuantizationFormat(Enum):
    # default format
    bf16 = "bf16"

    # used for enabling fp8_rowwise inference, some weights are bf16
    fp8_mixed = "fp8-mixed"

    int8 = "int8"

    int4 = "int4"


class ModelFamily(Enum):
    llama2 = "llama2"
    llama3 = "llama3"
    llama3_1 = "llama3_1"
    llama3_2 = "llama3_2"
    llama3_3 = "llama3_3"
    safety = "safety"


class CoreModelId(Enum):
    """Each of these models is a unique "SKU". These root models can be served in various garbs (especially by quantizing them)"""

    # Llama 2 family
    llama2_7b = "Llama-2-7b"
    llama2_13b = "Llama-2-13b"
    llama2_70b = "Llama-2-70b"
    llama2_7b_chat = "Llama-2-7b-chat"
    llama2_13b_chat = "Llama-2-13b-chat"
    llama2_70b_chat = "Llama-2-70b-chat"

    # Llama 3 family
    llama3_8b = "Llama-3-8B"
    llama3_70b = "Llama-3-70B"
    llama3_8b_instruct = "Llama-3-8B-Instruct"
    llama3_70b_instruct = "Llama-3-70B-Instruct"

    # Llama 3.1 family
    llama3_1_8b = "Llama3.1-8B"
    llama3_1_70b = "Llama3.1-70B"
    llama3_1_405b = "Llama3.1-405B"
    llama3_1_8b_instruct = "Llama3.1-8B-Instruct"
    llama3_1_70b_instruct = "Llama3.1-70B-Instruct"
    llama3_1_405b_instruct = "Llama3.1-405B-Instruct"

    # Llama 3.2 family
    llama3_2_1b = "Llama3.2-1B"
    llama3_2_3b = "Llama3.2-3B"
    llama3_2_1b_instruct = "Llama3.2-1B-Instruct"
    llama3_2_3b_instruct = "Llama3.2-3B-Instruct"
    llama3_2_11b_vision = "Llama3.2-11B-Vision"
    llama3_2_90b_vision = "Llama3.2-90B-Vision"
    llama3_2_11b_vision_instruct = "Llama3.2-11B-Vision-Instruct"
    llama3_2_90b_vision_instruct = "Llama3.2-90B-Vision-Instruct"

    # Llama 3.3 family
    llama3_3_70b_instruct = "Llama3.3-70B-Instruct"

    # Safety models
    llama_guard_3_8b = "Llama-Guard-3-8B"
    llama_guard_2_8b = "Llama-Guard-2-8B"
    llama_guard_3_11b_vision = "Llama-Guard-3-11B-Vision"
    llama_guard_3_1b = "Llama-Guard-3-1B"


def is_multimodal(model_id) -> bool:
    if model_id in [
        CoreModelId.llama3_2_11b_vision,
        CoreModelId.llama3_2_90b_vision,
        CoreModelId.llama3_2_11b_vision_instruct,
        CoreModelId.llama3_2_90b_vision_instruct,
    ]:
        return True
    else:
        return False


def model_family(model_id) -> ModelFamily:
    if model_id in [
        CoreModelId.llama2_7b,
        CoreModelId.llama2_13b,
        CoreModelId.llama2_70b,
        CoreModelId.llama2_7b_chat,
        CoreModelId.llama2_13b_chat,
        CoreModelId.llama2_70b_chat,
    ]:
        return ModelFamily.llama2
    elif model_id in [
        CoreModelId.llama3_8b,
        CoreModelId.llama3_70b,
        CoreModelId.llama3_8b_instruct,
        CoreModelId.llama3_70b_instruct,
    ]:
        return ModelFamily.llama3
    elif model_id in [
        CoreModelId.llama3_1_8b,
        CoreModelId.llama3_1_70b,
        CoreModelId.llama3_1_405b,
        CoreModelId.llama3_1_8b_instruct,
        CoreModelId.llama3_1_70b_instruct,
        CoreModelId.llama3_1_405b_instruct,
    ]:
        return ModelFamily.llama3_1
    elif model_id in [
        CoreModelId.llama3_2_1b,
        CoreModelId.llama3_2_3b,
        CoreModelId.llama3_2_1b_instruct,
        CoreModelId.llama3_2_3b_instruct,
        CoreModelId.llama3_2_11b_vision,
        CoreModelId.llama3_2_90b_vision,
        CoreModelId.llama3_2_11b_vision_instruct,
        CoreModelId.llama3_2_90b_vision_instruct,
    ]:
        return ModelFamily.llama3_2
    elif model_id in [
        CoreModelId.llama3_3_70b_instruct,
    ]:
        return ModelFamily.llama3_3
    elif model_id in [
        CoreModelId.llama_guard_3_8b,
        CoreModelId.llama_guard_2_8b,
        CoreModelId.llama_guard_3_11b_vision,
        CoreModelId.llama_guard_3_1b,
    ]:
        return ModelFamily.safety
    else:
        raise ValueError(f"Unknown model family for {model_id}")


class Model(BaseModel):
    core_model_id: CoreModelId
    description: str
    huggingface_repo: Optional[str] = None
    recommended_sampling_params: Optional[SamplingParams] = None
    arch_args: Dict[str, Any]
    variant: str = ""

    quantization_format: CheckpointQuantizationFormat = CheckpointQuantizationFormat.bf16
    pth_file_count: int
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # silence pydantic until we remove the `model_` fields
    model_config = ConfigDict(protected_namespaces=())

    @property
    def model_family(self) -> ModelFamily:
        return model_family(self.core_model_id)

    # The SKU is uniquely identified by (model_id, variant) combo
    def descriptor(self, shorten_default_variant: bool = True) -> str:
        if not self.variant:
            return self.core_model_id.value
        return f"{self.core_model_id.value}:{self.variant}"

    @property
    def is_instruct_model(self) -> bool:
        return "instruct" in self.id.name

    # Featured models are shown in the non-exhaustive model list
    @property
    def is_featured(self) -> bool:
        return self.model_family in [
            ModelFamily.llama3_1,
            ModelFamily.llama3_2,
            ModelFamily.llama3_3,
            ModelFamily.safety,
        ]

    @property
    def max_seq_length(self) -> int:
        if self.model_family == ModelFamily.llama2:
            return 4096
        elif self.core_model_id == CoreModelId.llama_guard_2_8b:
            return 4096
        elif self.model_family == ModelFamily.llama3:
            return 8192
        elif self.model_family in [ModelFamily.llama3_1, ModelFamily.llama3_3]:
            return 131072
        elif self.model_family == ModelFamily.llama3_2:
            if self.quantization_format == CheckpointQuantizationFormat.int4:
                return 8192
            return 131072
        elif self.core_model_id in [
            CoreModelId.llama_guard_3_8b,
            CoreModelId.llama_guard_3_11b_vision,
            CoreModelId.llama_guard_3_1b,
        ]:
            return 131072
        else:
            raise ValueError(f"Unknown max_seq_len for {self.core_model_id}")
