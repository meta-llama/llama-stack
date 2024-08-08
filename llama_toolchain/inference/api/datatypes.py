# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import List, Literal, Optional, Union

from llama_models.schema_utils import json_schema_type

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from llama_models.llama3_1.api.datatypes import *  # noqa: F403


class LogProbConfig(BaseModel):
    top_k: Optional[int] = 0


@json_schema_type
class QuantizationType(Enum):
    bf16 = "bf16"
    fp8 = "fp8"


@json_schema_type
class Fp8QuantizationConfig(BaseModel):
    type: Literal[QuantizationType.fp8.value] = QuantizationType.fp8.value


@json_schema_type
class Bf16QuantizationConfig(BaseModel):
    type: Literal[QuantizationType.bf16.value] = QuantizationType.bf16.value


QuantizationConfig = Annotated[
    Union[Bf16QuantizationConfig, Fp8QuantizationConfig],
    Field(discriminator="type"),
]


@json_schema_type
class ChatCompletionResponseEventType(Enum):
    start = "start"
    complete = "complete"
    progress = "progress"


@json_schema_type
class ToolCallParseStatus(Enum):
    started = "started"
    in_progress = "in_progress"
    failure = "failure"
    success = "success"


@json_schema_type
class ToolCallDelta(BaseModel):
    content: Union[str, ToolCall]
    parse_status: ToolCallParseStatus


@json_schema_type
class ChatCompletionResponseEvent(BaseModel):
    """Chat completion response event."""

    event_type: ChatCompletionResponseEventType
    delta: Union[str, ToolCallDelta]
    logprobs: Optional[List[TokenLogProbs]] = None
    stop_reason: Optional[StopReason] = None
