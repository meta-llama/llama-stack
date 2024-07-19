# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import collections
from enum import Enum, unique
from typing import Optional, Type

try:
    import fbgemm_gpu.experimental.gen_ai  # noqa: F401

    print("Using efficient FP8 operators in FBGEMM.")
except (ImportError, ModuleNotFoundError):
    print("No efficient FP8 operators. Please install FBGEMM in fp8_requirements.txt.")

import torch
from torch import nn, Tensor


@unique
class FfnQuantizeMode(Enum):
    FP8_ROWWISE = "fp8_rowwise"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


class Fp8ScaledWeights:
    # TODO: Ugly trick so torch allows us to replace parameters
    # with our custom Fp8Weights instance. Do this properly.
    @property
    def __class__(self) -> Type[nn.parameter.Parameter]:
        return nn.Parameter

    @property
    def grad_fn(self) -> None:
        return None


# pyre-fixme[4]: Attribute annotation cannot be `Any`.
# pyre-fixme[2]: Parameter annotation cannot be `Any`.
class Fp8RowwiseWeights(
    Fp8ScaledWeights,
    collections.namedtuple(
        "Fp8RowwiseWeights",
        ["weight", "scale", "shape", "activation_scale_ub"],
    ),
):
    pass


def ffn_swiglu(
    x: Tensor,
    w1: Fp8RowwiseWeights,
    w3: Fp8RowwiseWeights,
    w2: Fp8RowwiseWeights,
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    if (
        isinstance(w1, Fp8ScaledWeights)
        and isinstance(w3, Fp8ScaledWeights)
        and isinstance(w2, Fp8ScaledWeights)
    ):
        return ffn_swiglu_fp8_dynamic(
            x, w1, w3, w2, w1.activation_scale_ub, num_tokens, is_memory_bounded
        )

    (B, T, D) = x.shape
    (HD_L, D_) = w1.shape
    assert D_ == D

    assert isinstance(w1, Tensor)
    assert isinstance(w3, Tensor)
    x1 = x.view(B * T, D) @ w1.T
    x2 = x.view(B * T, D) @ w3.T
    z = torch.nn.functional.silu(x1) * x2
    del x1, x2
    assert isinstance(w2, Tensor)
    return (z @ w2.T).view(B, T, D)


@torch.inference_mode()
def quantize_fp8(
    w: Tensor,
    fp8_activation_scale_ub: float,
    mode: Optional[FfnQuantizeMode] = None,
    output_device: Optional[torch.device] = None,
) -> Fp8RowwiseWeights:
    """Quantize [n, k] weight tensor.

    Args:
        w (Tensor): [n, k] input high precision tensor to quantize.
        fp8_activation_scale_ub (float): Upper bound for activation max.
        mode (FfnQuantizeMode): Quantization mode.
    """
    activation_scale_ub = torch.tensor(
        [fp8_activation_scale_ub],
        dtype=torch.float,
        device="cuda",
    )
    if mode is not None and mode == FfnQuantizeMode.FP8_ROWWISE:  # rowwise
        wq, w_scale = torch.ops.fbgemm.quantize_fp8_per_row(w)
        del w
        return Fp8RowwiseWeights(
            weight=wq,
            scale=w_scale,
            shape=wq.shape,
            activation_scale_ub=activation_scale_ub,
        )


def fc_fp8_dynamic(
    x: Tensor,
    w: Fp8RowwiseWeights,
    activation_scale_ub: Optional[Tensor] = None,
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    """
    Single w8a8 fc layer with dynamic row-wise scaling.
    """
    if isinstance(w, Fp8RowwiseWeights):
        xq, x_scale = torch.ops.fbgemm.quantize_fp8_per_row(
            x, num_tokens, activation_scale_ub
        )
        y = torch.ops.fbgemm.f8f8bf16_rowwise(
            xq, w.weight, x_scale, w.scale, use_fast_accum=True
        )
    del xq
    return y


def ffn_swiglu_fp8_dynamic(
    x: Tensor,
    w1: Fp8RowwiseWeights,
    w3: Fp8RowwiseWeights,
    w2: Fp8RowwiseWeights,
    activation_scale_ub: Optional[Tensor] = None,
    num_tokens: Optional[Tensor] = None,
    is_memory_bounded: bool = False,
) -> Tensor:
    (B, T, D) = x.shape
    HD_L = w1.shape[0]
    assert HD_L == w3.shape[0]
    x1 = fc_fp8_dynamic(
        x.view(B * T, D),
        w1,
        activation_scale_ub,
        num_tokens,
        is_memory_bounded,
    )
    x2 = fc_fp8_dynamic(
        x.view(B * T, D),
        w3,
        activation_scale_ub,
        num_tokens,
        is_memory_bounded,
    )
    z = torch.nn.functional.silu(x1) * x2
    del x1, x2

    z_ = fc_fp8_dynamic(z, w2, activation_scale_ub, num_tokens, is_memory_bounded)

    return z_.view(B, T, D)
