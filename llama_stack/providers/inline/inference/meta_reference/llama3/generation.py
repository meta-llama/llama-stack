# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Generator, Optional, Union

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama_stack.apis.inference import (
    Fp8QuantizationConfig,
    Int4QuantizationConfig,
)
from llama_stack.log import get_logger
from llama_stack.models.llama.datatypes import Model
from llama_stack.models.llama.llama3.chat_format import ChatFormat, LLMInput
from llama_stack.models.llama.llama3.tokenizer import Tokenizer
from llama_stack.models.llama.sku_list import resolve_model

from ..common import TokenResult, model_checkpoint_dir
from ..config import MetaReferenceInferenceConfig, MetaReferenceQuantizedInferenceConfig
from .args import ModelArgs
from .model import Transformer
from .multimodal.model import CrossAttentionTransformer

log = get_logger(__name__, category="inference")


class Llama3:
    @staticmethod
    def build(
        config: Union[MetaReferenceInferenceConfig, MetaReferenceQuantizedInferenceConfig],
        model_id: str,
        llama_model: Model,
    ):
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
        if "DEVICE" in os.environ:
            device = os.environ.get("DEVICE")
            if device == "cuda":
                assert torch.cuda.is_available(), "PyTorch CUDA backend not available"
            if device == "xpu":
                assert torch.xpu.is_available(), "PyTorch XPU backend not available"
        else:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.xpu.is_available():
                device = "xpu"
            else:
                device = "cpu"
        log.info(f"Using {device} device")

        llama_model_id = llama_model.core_model_id.value
        if not torch.distributed.is_initialized():
            if device == "cuda":
                torch.distributed.init_process_group("nccl")
            else:
                torch.distributed.init_process_group("gloo")

        model_parallel_size = llama_model.pth_file_count

        if not model_parallel_is_initialized():
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device == "cuda":
            torch.cuda.set_device(local_rank)
        elif device == "xpu":
            torch.xpu.set_device(local_rank)

        # seed must be the same in all processes
        if config.torch_seed is not None:
            torch.manual_seed(config.torch_seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        if config.checkpoint_dir and config.checkpoint_dir != "null":
            ckpt_dir = config.checkpoint_dir
        else:
            resolved_model = resolve_model(model_id)
            if resolved_model is None:
                # if the model is not a native llama model, get the default checkpoint_dir based on model id
                ckpt_dir = model_checkpoint_dir(model_id)
            else:
                # if the model is a native llama model, get the default checkpoint_dir based on model core_model_id value
                ckpt_dir = model_checkpoint_dir(resolved_model.descriptor())

        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(checkpoints), (
            f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        )
        ckpt_path = checkpoints[get_model_parallel_rank()]
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        if "model" in params:
            params = params["model"]

        model_args: ModelArgs = ModelArgs(
            max_seq_len=config.max_seq_len,
            max_batch_size=config.max_batch_size,
            **params,
        )

        tokenizer = Tokenizer.get_instance()
        assert model_args.vocab_size == tokenizer.n_words, (
            f"model_args vocab = {model_args.vocab_size} but tokenizer vocab = {tokenizer.n_words}"
        )

        if isinstance(config, MetaReferenceQuantizedInferenceConfig):
            if isinstance(config.quantization, Fp8QuantizationConfig):
                from .quantization.loader import convert_to_fp8_quantized_model

                # load on CPU in bf16 so that fp8 conversion does not find an
                # unexpected (fp32, e.g.) datatype
                torch.set_default_tensor_type(torch.BFloat16Tensor)
                if model_args.vision_chunk_size > 0:
                    model = CrossAttentionTransformer(model_args)
                    model.setup_cache(model_args.max_batch_size, torch.bfloat16)
                else:
                    model = Transformer(model_args)
                model.load_state_dict(state_dict, strict=False)
                model = convert_to_fp8_quantized_model(model, config, ckpt_dir)
            elif isinstance(config.quantization, Int4QuantizationConfig):
                from .quantization.loader import convert_to_int4_quantized_model

                model = Transformer(model_args)
                model = convert_to_int4_quantized_model(model, model_args, config)
                model.load_state_dict(state_dict, strict=True)

                if model_args.quantization_args is not None and model_args.quantization_args.spinquant:
                    # Add a wrapper for adding hadamard transform for spinquant.
                    # This needs to be done after loading the state dict otherwise an error will be raised while
                    # loading the state dict.
                    from ..hadamard_utils import (
                        add_hadamard_transform_for_spinquant,
                    )

                    add_hadamard_transform_for_spinquant(model)
            else:
                raise NotImplementedError("Currently int4 and fp8 are the only supported quantization methods.")
        else:
            if device == "cuda":
                if torch.cuda.is_bf16_supported():
                    torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
                else:
                    torch.set_default_tensor_type(torch.cuda.HalfTensor)
            else:
                torch.set_default_device(device)
                if device == "xpu" and torch.xpu.is_bf16_supported():
                    torch.set_default_dtype(torch.bfloat16)
                else:
                    torch.set_default_dtype(torch.half)
            if model_args.vision_chunk_size > 0:
                model = CrossAttentionTransformer(model_args)
                model.setup_cache(model_args.max_batch_size, torch.bfloat16)
            else:
                model = Transformer(model_args)
            model.load_state_dict(state_dict, strict=False)

        model.to(device)

        log.info(f"Loaded in {time.time() - start_time:.2f} seconds")
        return Llama3(model, tokenizer, model_args, llama_model_id)

    def __init__(
        self,
        model: Transformer,
        tokenizer: Tokenizer,
        args: ModelArgs,
        llama_model: str,
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)
        self.llama_model = llama_model

    @torch.inference_mode()
    def generate(
        self,
        model_input: LLMInput,
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        print_input_tokens: bool = False,
        logits_processor: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> Generator:
        params = self.model.params

        if print_input_tokens:
            input_tokens = [self.formatter.vision_token if t == 128256 else t for t in model_input.tokens]
            log.info("Input to model -> " + self.tokenizer.decode(input_tokens))
        prompt_tokens = [model_input.tokens]

        bsz = 1
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)

        if max_prompt_len >= params.max_seq_len:
            log.error(f"Out of token budget {max_prompt_len} vs {params.max_seq_len}")
            return

        total_len = min(max_gen_len + max_prompt_len, params.max_seq_len)

        is_vision = isinstance(self.model, CrossAttentionTransformer)
        if is_vision:
            images = model_input.vision.images if model_input.vision is not None else []
            mask = model_input.vision.mask if model_input.vision is not None else []

            # the method works for bsz > 1 so add a batch dimension
            xattn_caches, cross_attention_masks, full_text_row_masked_out_mask = self.model.compute_vision_tokens_masks(
                batch_images=[images],
                batch_masks=[mask],
                total_len=total_len,
            )

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz)
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            # TODO(ashwin): unify this branch with the one below and figure out multimodal crap
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(self.tokenizer.stop_tokens)
        for cur_pos in range(min_prompt_len, total_len):
            if is_vision:
                position_ids = torch.arange(prev_pos, cur_pos, dtype=torch.long)
                logits = self.model.forward(
                    position_ids,
                    tokens,
                    cross_attention_masks,
                    full_text_row_masked_out_mask,
                    xattn_caches,
                )
            else:
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if logits_processor is not None:
                logits = logits_processor(tokens[:, :cur_pos], logits)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            target = tokens[:, prev_pos + 1 : cur_pos + 1]
            if is_vision:
                # the logits space (num_classes) is designed to never contain a media_token
                # however our input token stream does contain them. we need to nuke them here
                # or else the CUDA kernels will crash with an illegal memory access
                vision_tokens = [self.tokenizer.special_tokens["<|image|>"], 128256]
                masks = [target.eq(t) for t in vision_tokens]
                if len(masks) > 1:
                    mask = torch.logical_or(*masks)
                else:
                    mask = masks[0]
                target[mask] = 0

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (torch.isin(next_token, stop_tokens))
            yield TokenResult(
                token=next_token[0].item(),
                text=self.tokenizer.decode(next_token.tolist()),
                logprobs=(token_logprobs[:, cur_pos : cur_pos + 1][0].tolist() if logprobs else None),
            )

            prev_pos = cur_pos
            if all(eos_reached):
                break


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
