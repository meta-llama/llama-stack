# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import fire

import torch
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from fp8.fp8_impls import FfnQuantizeMode, quantize_fp8

from llama.model import ModelArgs, Transformer, TransformerBlock
from llama.tokenizer import Tokenizer
from torch.nn.parameter import Parameter


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    quantized_ckpt_dir: str,
    max_seq_len: Optional[int] = 512,
    max_batch_size: Optional[int] = 4,
    model_parallel_size: Optional[int] = None,
    ffn_quantize_mode: Optional[FfnQuantizeMode] = FfnQuantizeMode.FP8_ROWWISE,
    fp8_activation_scale_ub: Optional[float] = 1200.0,
    seed: int = 1,
):
    """ """
    if not os.path.exists(quantized_ckpt_dir):
        os.makedirs(quantized_ckpt_dir)
        shutil.copy(
            os.path.join(ckpt_dir, "params.json"),
            os.path.join(quantized_ckpt_dir, "params.json"),
        )
        shutil.copy(
            os.path.join(ckpt_dir, "tokenizer.model"),
            os.path.join(quantized_ckpt_dir, "tokenizer.model"),
        )

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert (
            model_args.vocab_size == tokenizer.n_words
        ), f"model_args vocab = {model_args.vocab_size} but tokenizer vocab = {tokenizer.n_words}"

        # load on CPU in bf16 so that fp8 conversion does not find an unexpected (fp32, e.g.) datatype
        torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)

        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        print(ckpt_path)
        assert (
            quantized_ckpt_dir is not None
        ), "QUantized checkpoint directory should not be None"
        fp8_scales = {}
        for block in model.layers:
            if isinstance(block, TransformerBlock):
                if block.layer_id == 0 or block.layer_id == (model.n_layers - 1):
                    continue

                fp8_weight = quantize_fp8(
                    block.feed_forward.w1.weight,
                    fp8_activation_scale_ub,
                    ffn_quantize_mode,
                    output_device=torch.device("cpu"),
                )
                with torch.inference_mode():
                    block.feed_forward.w1.weight = Parameter(fp8_weight.weight)
                fp8_scales[
                    f"{block.layer_id}_feed_forward.w1_{get_model_parallel_rank()}"
                ] = fp8_weight.scale

                fp8_weight = quantize_fp8(
                    block.feed_forward.w3.weight,
                    fp8_activation_scale_ub,
                    ffn_quantize_mode,
                    output_device=torch.device("cpu"),
                )
                with torch.inference_mode():
                    block.feed_forward.w3.weight = Parameter(fp8_weight.weight)
                fp8_scales[
                    f"{block.layer_id}_feed_forward.w3_{get_model_parallel_rank()}"
                ] = fp8_weight.scale

                fp8_weight = quantize_fp8(
                    block.feed_forward.w2.weight,
                    fp8_activation_scale_ub,
                    ffn_quantize_mode,
                    output_device=torch.device("cpu"),
                )
                with torch.inference_mode():
                    block.feed_forward.w2.weight = Parameter(fp8_weight.weight)
                fp8_scales[
                    f"{block.layer_id}_feed_forward.w2_{get_model_parallel_rank()}"
                ] = fp8_weight.scale

        fp8_scales_path = os.path.join(
            quantized_ckpt_dir, f"fp8_scales_{get_model_parallel_rank()}.pt"
        )
        torch.save(fp8_scales, fp8_scales_path)

        ckpt_path = os.path.join(
            quantized_ckpt_dir,
            "consolidated.{:02d}.pth".format(get_model_parallel_rank()),
        )
        torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    fire.Fire(main)
