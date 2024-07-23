# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

from hydra_zen import builds
from hydra.core.config_store import ConfigStore
from llama_models.llama3_1.api.datatypes import CheckpointQuantizationFormat

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from .datatypes import QuantizationConfig


class ImplType(Enum):
    inline = "inline"
    remote = "remote"


class CheckpointType(Enum):
    pytorch = "pytorch"
    huggingface = "huggingface"


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


class HuggingFaceCheckpoint(BaseModel):
    checkpoint_type: Literal[CheckpointType.huggingface.value] = (
        CheckpointType.huggingface.value
    )
    repo_id: str  # or model_name ?
    model_parallel_size: int
    quantization_format: CheckpointQuantizationFormat = (
        CheckpointQuantizationFormat.bf16
    )


class ModelCheckpointConfig(BaseModel):
    checkpoint: Annotated[
        Union[PytorchCheckpoint, HuggingFaceCheckpoint],
        Field(discriminator="checkpoint_type"),
    ]


class InlineImplConfig(BaseModel):
    impl_type: Literal[ImplType.inline.value] = ImplType.inline.value
    checkpoint_config: ModelCheckpointConfig
    quantization: Optional[QuantizationConfig] = None
    torch_seed: Optional[int] = None
    max_seq_len: int
    max_batch_size: int = 1


class RemoteImplConfig(BaseModel):
    impl_type: Literal[ImplType.remote.value] = ImplType.remote.value
    url: str = Field(..., description="The URL of the remote module")


class InferenceConfig(BaseModel):
    impl_config: Annotated[
        Union[InlineImplConfig, RemoteImplConfig],
        Field(discriminator="impl_type"),
    ]


InferenceHydraConfig = builds(InferenceConfig)

cs = ConfigStore.instance()
cs.store(name="inference_config", node=InferenceHydraConfig)
