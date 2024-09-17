# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from typing import Optional

import torch

from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region
from llama_models.llama3.api.model import Transformer, TransformerBlock
from llama_stack.apis.inference import QuantizationType

from llama_stack.apis.inference.config import (
    CheckpointQuantizationFormat,
    MetaReferenceImplConfig,
)

from termcolor import cprint
from torch import Tensor


def is_fbgemm_available() -> bool:
    try:
        import fbgemm_gpu.experimental.gen_ai  # noqa: F401

        return True
    except ImportError:
        return False


def swiglu_wrapper(
    self,
    x: Tensor,
):
    from .fp8_impls import ffn_swiglu

    out = ffn_swiglu(x, self.w1.weight, self.w3.weight, self.w2.weight)
    return reduce_from_model_parallel_region(out)


def convert_to_quantized_model(
    model: Transformer,
    config: MetaReferenceImplConfig,
    fp8_activation_scale_ub: Optional[float] = 1200.0,
) -> Transformer:
    if config.quantization.type == QuantizationType.bf16.value:
        return model

    elif config.quantization.type != QuantizationType.fp8.value:
        raise ValueError("Only FP8 quantization is supported")

    from .fp8_impls import Fp8ScaledWeights, load_fp8, quantize_fp8

    checkpoint = config.checkpoint_config.checkpoint
    # Move weights to GPU with quantization
    if checkpoint.quantization_format == CheckpointQuantizationFormat.fp8_mixed.value:
        cprint("Loading fp8 scales...", "yellow")
        fp8_scales_path = os.path.join(
            checkpoint.checkpoint_dir, f"fp8_scales_{get_model_parallel_rank()}.pt"
        )
        assert os.path.isfile(
            fp8_scales_path
        ), f"fp8_scales_path not found for rank {get_model_parallel_rank()}"
        fp8_scales = torch.load(fp8_scales_path, weights_only=True)

        for block in model.layers:
            if isinstance(block, TransformerBlock):
                if block.layer_id == 0 or block.layer_id == (model.n_layers - 1):
                    continue

                block.feed_forward.forward = swiglu_wrapper.__get__(block.feed_forward)
                for key in ("w1", "w3", "w2"):
                    param = getattr(block.feed_forward, key)
                    param.weight = load_fp8(
                        param.weight,
                        fp8_scales[
                            f"{block.layer_id}_feed_forward.{key}_{get_model_parallel_rank()}"
                        ],
                        fp8_activation_scale_ub,
                    )
    else:
        cprint("Quantizing fp8 weights from bf16...", "yellow")
        for block in model.layers:
            if isinstance(block, TransformerBlock):
                if block.layer_id == 0 or block.layer_id == (model.n_layers - 1):
                    continue
                block.feed_forward.forward = swiglu_wrapper.__get__(block.feed_forward)
                for key in ("w1", "w3", "w2"):
                    param = getattr(block.feed_forward, key)
                    param.weight = quantize_fp8(
                        param.weight,
                        fp8_activation_scale_ub,
                        output_device=torch.device("cuda"),
                    )

    for _, parameter in model.named_parameters():
        if not isinstance(parameter, Fp8ScaledWeights):
            parameter.data = parameter.to(device="cuda")
    return model
