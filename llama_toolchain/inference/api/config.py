# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Literal, Optional, Union

from hydra.core.config_store import ConfigStore

from hydra_zen import builds
from llama_models.llama3_1.api.datatypes import CheckpointQuantizationFormat

from pydantic import BaseModel, Field
from strong_typing.schema import json_schema_type
from typing_extensions import Annotated

from .datatypes import QuantizationConfig


@json_schema_type
class ImplType(Enum):
    inline = "inline"
    remote = "remote"
    ollama = "ollama"


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
class InlineImplConfig(BaseModel):
    impl_type: Literal[ImplType.inline.value] = ImplType.inline.value
    checkpoint_config: ModelCheckpointConfig
    quantization: Optional[QuantizationConfig] = None
    torch_seed: Optional[int] = None
    max_seq_len: int
    max_batch_size: int = 1


@json_schema_type
class RemoteImplConfig(BaseModel):
    impl_type: Literal[ImplType.remote.value] = ImplType.remote.value
    url: str = Field(..., description="The URL of the remote module")


@json_schema_type
class OllamaImplConfig(BaseModel):
    impl_type: Literal[ImplType.ollama.value] = ImplType.ollama.value
    model: str = Field(..., description="The name of the model in ollama catalog")
    url: str = Field(..., description="The URL for the ollama server")


@json_schema_type
class InferenceConfig(BaseModel):
    impl_config: Annotated[
        Union[InlineImplConfig, RemoteImplConfig, OllamaImplConfig],
        Field(discriminator="impl_type"),
    ]


InferenceHydraConfig = builds(InferenceConfig)

cs = ConfigStore.instance()
cs.store(name="inference_config", node=InferenceHydraConfig)
