# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from fp8.fp8_impls import (
    FfnQuantizeMode,
    Fp8ScaledWeights,
    load_fp8,
    ModelLoadMode,
    quantize_fp8,
)

from llama.model import ModelArgs, Transformer, TransformerBlock
from llama.tokenizer import ChatFormat, Dialog, Message, ModelInput, Tokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        ffn_quantize_mode: Optional[FfnQuantizeMode] = FfnQuantizeMode.NONE,
        model_load_mode: Optional[ModelLoadMode] = ModelLoadMode.BF16,
        fp8_activation_scale_ub: Optional[float] = 1200.0,
        seed: int = 1,
    ) -> "Llama":
        """
        Build a Llama instance by initializing and loading a model checkpoint.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.
        """
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

        start_time = time.time()
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

        print("ffn_quantize_mode: ", ffn_quantize_mode)
        if ffn_quantize_mode == FfnQuantizeMode.FP8_ROWWISE:
            # Move weights to GPU with quantization
            if model_load_mode == ModelLoadMode.FP8:
                fp8_scales_path = os.path.join(
                    ckpt_dir, f"fp8_scales_{get_model_parallel_rank()}.pt"
                )
                assert os.path.isfile(
                    fp8_scales_path
                ), f"fp8_scales_path not found for rank {get_model_parallel_rank()}"
                fp8_scales = torch.load(fp8_scales_path, weights_only=True)

                for block in model.layers:
                    if isinstance(block, TransformerBlock):
                        if block.layer_id == 0 or block.layer_id == (
                            model.n_layers - 1
                        ):
                            continue
                        block.feed_forward.w1.weight = load_fp8(
                            block.feed_forward.w1.weight,
                            fp8_scales[
                                f"{block.layer_id}_feed_forward.w1_{get_model_parallel_rank()}"
                            ],
                            fp8_activation_scale_ub,
                        )
                        block.feed_forward.w3.weight = load_fp8(
                            block.feed_forward.w3.weight,
                            fp8_scales[
                                f"{block.layer_id}_feed_forward.w3_{get_model_parallel_rank()}"
                            ],
                            fp8_activation_scale_ub,
                        )
                        block.feed_forward.w2.weight = load_fp8(
                            block.feed_forward.w2.weight,
                            fp8_scales[
                                f"{block.layer_id}_feed_forward.w2_{get_model_parallel_rank()}"
                            ],
                            fp8_activation_scale_ub,
                        )
            else:
                for block in model.layers:
                    if isinstance(block, TransformerBlock):
                        if block.layer_id == 0 or block.layer_id == (
                            model.n_layers - 1
                        ):
                            continue
                        block.feed_forward.w1.weight = quantize_fp8(
                            block.feed_forward.w1.weight,
                            fp8_activation_scale_ub,
                            ffn_quantize_mode,
                            output_device=torch.device("cuda"),
                        )
                        block.feed_forward.w3.weight = quantize_fp8(
                            block.feed_forward.w3.weight,
                            fp8_activation_scale_ub,
                            ffn_quantize_mode,
                            output_device=torch.device("cuda"),
                        )
                        block.feed_forward.w2.weight = quantize_fp8(
                            block.feed_forward.w2.weight,
                            fp8_activation_scale_ub,
                            ffn_quantize_mode,
                            output_device=torch.device("cuda"),
                        )

            for _, parameter in model.named_parameters():
                if not isinstance(parameter, Fp8ScaledWeights):
                    parameter.data = parameter.to(device="cuda")
        else:
            for _, parameter in model.named_parameters():
                parameter.data = parameter.to(device="cuda")

        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer, model_args)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, args: ModelArgs):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)

    @torch.inference_mode()
    def generate(
        self,
        model_inputs: List[ModelInput],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
        include_stop_token: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        prompt_tokens = [m.tokens for m in model_inputs]
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens))

        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token

            target = tokens[:, prev_pos + 1 : cur_pos + 1]

            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                torch.isin(next_token, stop_tokens)
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            for stop_token in self.tokenizer.stop_tokens:
                try:
                    eos_idx = toks.index(stop_token)
                    if include_stop_token:
                        eos_idx += 1
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError:
                    pass
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            model_inputs=[ModelInput(tokens=pt) for pt in prompt_tokens],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.
        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1

        model_inputs = [
            self.formatter.encode_dialog_prompt(dialog) for dialog in dialogs
        ]
        generation_tokens, generation_logprobs = self.generate(
            model_inputs=model_inputs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            include_stop_token=True,
        )
        if logprobs:
            return [
                {
                    "generation": self.formatter.decode_assistant_message(t),
                    "tokens": [self.tokenizer.decode([x]) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {
                "generation": self.formatter.decode_assistant_message(t),
            }
            for t in generation_tokens
        ]


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
