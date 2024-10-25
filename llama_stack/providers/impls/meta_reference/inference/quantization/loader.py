# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import os
from typing import Any, Dict, List, Optional

import torch

from fairscale.nn.model_parallel.layers import ColumnParallelLinear, RowParallelLinear
from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region

from llama_models.datatypes import CheckpointQuantizationFormat

from llama_models.llama3.api.args import ModelArgs
from llama_models.llama3.reference_impl.model import Transformer, TransformerBlock
from llama_models.sku_list import resolve_model
from termcolor import cprint
from torch import nn, Tensor

from torchao.quantization.GPTQ import Int8DynActInt4WeightLinear

from llama_stack.apis.inference import QuantizationType

from llama_stack.providers.impls.meta_reference.inference.config import (
    MetaReferenceQuantizedInferenceConfig,
)


def swiglu_wrapper(
    self,
    x: Tensor,
):
    from .fp8_impls import ffn_swiglu

    out = ffn_swiglu(x, self.w1.weight, self.w3.weight, self.w2.weight)
    return reduce_from_model_parallel_region(out)


def convert_to_fp8_quantized_model(
    model: Transformer,
    config: MetaReferenceQuantizedInferenceConfig,
    checkpoint_dir: str,
    fp8_activation_scale_ub: Optional[float] = 1200.0,
) -> Transformer:
    if config.quantization.type == QuantizationType.bf16.value:
        return model

    elif config.quantization.type != QuantizationType.fp8.value:
        raise ValueError("Only FP8 quantization is supported")

    from .fp8_impls import Fp8ScaledWeights, load_fp8, quantize_fp8

    llama_model = resolve_model(config.model)
    assert llama_model is not None, f"Model {config.model} not found"

    # Move weights to GPU with quantization
    if llama_model.quantization_format == CheckpointQuantizationFormat.fp8_mixed.value:
        cprint("Loading fp8 scales...", "yellow")
        fp8_scales_path = os.path.join(
            checkpoint_dir, f"fp8_scales_{get_model_parallel_rank()}.pt"
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


class Int8DynActInt4WeightLinearLoRA(Int8DynActInt4WeightLinear):
    """
    Int8DynActInt4WeightLinear with LoRA adaptor.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to use bias.
        device: Device to use.
        group_size: Group size for quantization.
        precision: Precision of quantization.
        scales_precision: Precision of scales.
        lora_rank: Rank of LoRA adaptor.
        lora_scale: Scale of LoRA adaptor.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias=False,
        device=None,
        # quantization parameters
        group_size: int = 256,
        precision: torch.dtype = torch.float32,
        scales_precision: torch.dtype = torch.float32,
        # LoRA parameters
        lora_rank: Optional[int] = None,
        lora_scale: Optional[float] = None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            groupsize=group_size,
            precision=precision,
            scales_precision=scales_precision,
        )
        if lora_rank is not None:
            assert lora_scale is not None, "Please specify lora scale for LoRA."
            # Low-rank adaptation. See paper for more details: https://arxiv.org/abs/2106.09685
            self.adaptor = nn.Sequential()
            self.adaptor.add_module("A", nn.Linear(in_features, lora_rank, bias=False))
            self.adaptor.add_module("B", nn.Linear(lora_rank, out_features, bias=False))
            self.lora_scale = lora_scale
        else:
            self.adaptor = None
            self.lora_scale = None
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """A hook to load the quantized weights from the state dict."""
        if prefix + "zeros" not in state_dict:
            # Zero-point may not be saved in the state dict. In this case, we assume it's zero.
            assert prefix + "scales" in state_dict
            state_dict[prefix + "zeros"] = torch.zeros_like(
                state_dict[prefix + "scales"]
            )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        module_out = super().forward(input_)
        if self.adaptor is not None:
            adaptor_out = self.adaptor(input_) * self.lora_scale
            return module_out + adaptor_out
        return module_out


class Int8WeightEmbedding(torch.nn.Embedding):
    """An embedding layer to load int8 weights.

    Args:
        num_embeddings: Number of embeddings.
        embedding_dim: Embedding dimension.
        padding_idx: Padding index.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        device=None,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, device=device)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """A hook to load the quantized embedding weight and scales from the state dict."""
        weights = state_dict.pop(prefix + "weight")
        scales = state_dict.pop(prefix + "scales")
        state_dict[prefix + "weight"] = weights * scales


class Int8WeightLinear(torch.nn.Linear):
    """A linear layer to load int8 weights.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to use bias.
    """

    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device=device)

        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """A hook to load the quantized linear weight and scales from the state dict."""
        weights = state_dict.pop(prefix + "weight")
        scales = state_dict.pop(prefix + "scales")
        state_dict[prefix + "weight"] = weights * scales


def _prepare_model_int4_weight_int8_dynamic_activation(
    model: torch.nn.Module,
    group_size: int,
    lora_rank: Optional[int],
    lora_scale: Optional[float],
):
    """Prepare the model for int4 weight and int8 dynamic activation quantization.

    Note that the weights of embedding and output layers are quantized to int8.
    """
    device = None
    for module_name, module in model.named_children():
        if module_name == "output":
            quantized_module = Int8WeightLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias,
                device=device,
            )
            del module
            setattr(model, module_name, quantized_module)
        elif module_name == "tok_embeddings":
            quantized_module = Int8WeightEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                padding_idx=module.padding_idx,
                device=device,
            )
            del module
            setattr(model, module_name, quantized_module)
        elif isinstance(module, (ColumnParallelLinear, RowParallelLinear, nn.Linear)):
            quantized_module = Int8DynActInt4WeightLinearLoRA(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=False,
                group_size=group_size,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                device=device,
            )
            del module
            setattr(model, module_name, quantized_module)
        else:
            _prepare_model_int4_weight_int8_dynamic_activation(
                module, group_size, lora_rank, lora_scale
            )

    return model


def convert_to_int4_quantized_model(
    model: Transformer,
    model_args: ModelArgs,
    config: MetaReferenceQuantizedInferenceConfig,
) -> Transformer:
    """Convert the model to int4 quantized model."""

    if model_args.quantization_args is None:
        raise ValueError("'quantization_args' cannot be None. Please specify it.")

    quantization_args = model_args.quantization_args

    if quantization_args.scheme.value != "int4_weight_int8_dynamic_activation":
        raise NotImplementedError(
            "Only int4 quantization with 'int4_weight_int8_dynamic_activation' scheme is supported."
        )

    group_size = model_args.quantization_args.group_size
    if group_size is None:
        raise ValueError(
            "'group_size' cannot be None in 'quantization_args'. Please specify it."
        )

    if model_args.lora_args is None:
        # Certain quantized models (e.g., SpinQuant) may not have LoRA.
        lora_rank = None
        lora_scale = None
    else:
        lora_rank = model_args.lora_args.rank
        lora_scale = model_args.lora_args.scale

    _prepare_model_int4_weight_int8_dynamic_activation(
        model, group_size, lora_rank, lora_scale
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return model.to(device)
