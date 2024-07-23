# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Union

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


# Hydra does not like unions of containers and
# Pydantic does not like Literals
# Adding a simple dataclass with custom coversion
# to config classes


@dataclass
class InlineImplHydraConfig:
    checkpoint_type: str  # "pytorch" / "HF"
    # pytorch checkpoint required args
    checkpoint_dir: str
    tokenizer_path: str
    model_parallel_size: int
    max_seq_len: int
    max_batch_size: int = 1
    quantization: Optional[QuantizationConfig] = None
    # TODO: huggingface checkpoint required args

    def convert_to_inline_impl_config(self):
        if self.checkpoint_type == "pytorch":
            return InlineImplConfig(
                checkpoint_config=ModelCheckpointConfig(
                    checkpoint=PytorchCheckpoint(
                        checkpoint_type=CheckpointType.pytorch.value,
                        checkpoint_dir=self.checkpoint_dir,
                        tokenizer_path=self.tokenizer_path,
                        model_parallel_size=self.model_parallel_size,
                    )
                ),
                quantization=self.quantization,
                max_seq_len=self.max_seq_len,
                max_batch_size=self.max_batch_size,
            )
        else:
            raise NotImplementedError("HF Checkpoint not supported yet")


@dataclass
class RemoteImplHydraConfig:
    url: str

    def convert_to_remote_impl_config(self):
        return RemoteImplConfig(
            url=self.url,
        )


@dataclass
class InferenceHydraConfig:
    impl_type: str
    inline_config: Optional[InlineImplHydraConfig] = None
    remote_config: Optional[RemoteImplHydraConfig] = None

    def __post_init__(self):
        assert self.impl_type in ["inline", "remote"]
        if self.impl_type == "inline":
            assert self.inline_config is not None
        if self.impl_type == "remote":
            assert self.remote_config is not None

    def convert_to_inference_config(self):
        if self.impl_type == "inline":
            inline_config = InlineImplHydraConfig(**self.inline_config)
            return InferenceConfig(
                impl_config=inline_config.convert_to_inline_impl_config()
            )
        elif self.impl_type == "remote":
            remote_config = RemoteImplHydraConfig(**self.remote_config)
            return InferenceConfig(
                impl_config=remote_config.convert_to_remote_impl_config()
            )


cs = ConfigStore.instance()
cs.store(name="inference_config", node=InferenceHydraConfig)
