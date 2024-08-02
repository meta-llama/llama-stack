# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Literal, Optional, Union

from llama_models.llama3_1.api.datatypes import CheckpointQuantizationFormat

from pydantic import BaseModel, Field
from strong_typing.schema import json_schema_type
from typing_extensions import Annotated

from .datatypes import QuantizationConfig


@json_schema_type
class CheckpointType(Enum):
    pytorch = "pytorch"
    huggingface = "huggingface"


@json_schema_type
class PytorchCheckpoint(BaseModel):
    checkpoint_type: Literal[CheckpointType.pytorch.value] = (
        CheckpointType.pytorch.value
    )
    checkpoint_dir: str
    tokenizer_path: str
    model_parallel_size: int
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )


@json_schema_type
class HuggingFaceCheckpoint(BaseModel):
    checkpoint_type: Literal[CheckpointType.huggingface.value] = (
        CheckpointType.huggingface.value
    )
    repo_id: str  # or model_name ?
    model_parallel_size: int
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )


@json_schema_type
class ModelCheckpointConfig(BaseModel):
    checkpoint: Annotated[
        Union[PytorchCheckpoint, HuggingFaceCheckpoint],
        Field(discriminator="checkpoint_type"),
    ]


@json_schema_type
class MetaReferenceImplConfig(BaseModel):
    model: str
    checkpoint_config: ModelCheckpointConfig
    quantization: Optional[QuantizationConfig] = None
    torch_seed: Optional[int] = None
    max_seq_len: int
    max_batch_size: int = 1


@json_schema_type
class OllamaImplConfig(BaseModel):
    model: str = Field(..., description="The name of the model in ollama catalog")
    url: str = Field(..., description="The URL for the ollama server")
