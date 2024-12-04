# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, IAny, nc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Callable, Dict

import torch
from llama_models.sku_list import resolve_model

from torchtune.models.llama3 import llama3_tokenizer, lora_llama3_8b
from torchtune.models.llama3._tokenizer import Llama3Tokenizer
from torchtune.models.llama3_2 import lora_llama3_2_3b

LORA_MODEL_TYPES: Dict[str, Any] = {
    "Llama3.2-3B-Instruct": lora_llama3_2_3b,
    "Llama-3-8B-Instruct": lora_llama3_8b,
}

TOKENIZER_TYPES: Dict[str, Any] = {
    "Llama3.2-3B-Instruct": llama3_tokenizer,
    "Llama-3-8B-Instruct": llama3_tokenizer,
}

CHECKPOINT_MODEL_TYPES: Dict[str, str] = {
    "Llama3.2-3B-Instruct": "LLAMA3_2",
}

BuildLoraModelCallable = Callable[..., torch.nn.Module]
BuildTokenizerCallable = Callable[..., Llama3Tokenizer]


def get_model_type(
    model_id: str,
) -> BuildLoraModelCallable:
    model = resolve_model(model_id)
    return LORA_MODEL_TYPES[model.core_model_id.value]


def get_tokenizer_type(
    model_id: str,
) -> BuildTokenizerCallable:
    model = resolve_model(model_id)
    return TOKENIZER_TYPES[model.core_model_id.value]


def get_checkpointer_model_type(
    model_id: str,
) -> str:
    model = resolve_model(model_id)
    return CHECKPOINT_MODEL_TYPES[model.core_model_id.value]
