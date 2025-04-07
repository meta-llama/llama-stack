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

from enum import Enum
from typing import Optional

from pydantic import BaseModel, model_validator


class QuantizationScheme(Enum):
    int4_weight_int8_dynamic_activation = "int4_weight_int8_dynamic_activation"


class QuantizationArgs(BaseModel):
    scheme: Optional[QuantizationScheme] = None
    group_size: Optional[int] = None
    spinquant: bool = False


class LoRAArgs(BaseModel):
    rank: int
    scale: float


class MoEArgs(BaseModel):
    num_experts: int = -1
    capacity_factor: float = 1.0  # capacity factor determines how many tokens each expert can choose
    auto_scale_F: bool = (  # noqa: N815
        True  # if true, rescales hidden_dim such that number of activated params is same as equivalent dense layer
    )
    top_k: int = 1
    interleave_moe_layer_step: int = 1


class Size(BaseModel):
    height: int
    width: int


class VisionArgs(BaseModel):
    image_size: Size
    patch_size: Size

    # parameters for the encoder transformer
    dim: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    output_dim: int

    pixel_shuffle_ratio: float


class ModelArgs(BaseModel):
    dim: int = -1
    n_layers: int = -1
    n_heads: int = -1
    n_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    ffn_exp: Optional[float] = None
    norm_eps: float = 1e-5

    attention_chunk_size: Optional[int] = None
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    nope_layer_interval: Optional[int] = None  # No position encoding in every n layers
    use_qk_norm: bool = False
    # Set to True to enable inference-time temperature tuning (useful for very long context)
    attn_temperature_tuning: bool = False
    floor_scale: float = 8192.0
    attn_scale: float = 0.1

    vision_args: Optional[VisionArgs] = None
    moe_args: Optional[MoEArgs] = None
    quantization_args: Optional[QuantizationArgs] = None
    lora_args: Optional[LoRAArgs] = None

    max_batch_size: int = 32
    max_seq_len: int = 2048

    @model_validator(mode="after")
    def validate(self) -> "ModelArgs":
        assert self.n_kv_heads <= self.n_heads, f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.dim % self.n_heads == 0, f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        return self
