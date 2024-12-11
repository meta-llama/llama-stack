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

from enum import Enum
from typing import Any, Callable, Dict, List

import torch
from llama_stack.apis.datasets import Datasets
from llama_stack.apis.common.type_system import *  # noqa
from llama_models.sku_list import resolve_model
from llama_stack.apis.common.type_system import ParamType

from torchtune.models.llama3 import llama3_tokenizer, lora_llama3_8b
from torchtune.models.llama3._tokenizer import Llama3Tokenizer
from torchtune.models.llama3_2 import lora_llama3_2_3b


class ColumnName(Enum):
    instruction = "instruction"
    input = "input"
    output = "output"
    text = "text"


MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Llama3.2-3B-Instruct": {
        "model_definition": lora_llama3_2_3b,
        "tokenizer_type": llama3_tokenizer,
        "checkpoint_type": "LLAMA3_2",
    },
    "Llama-3-8B-Instruct": {
        "model_definition": lora_llama3_8b,
        "tokenizer_type": llama3_tokenizer,
        "checkpoint_type": "LLAMA3",
    },
}

EXPECTED_DATASET_SCHEMA: Dict[str, List[Dict[str, ParamType]]] = {
    "alpaca": [
        {
            ColumnName.instruction.value: StringType(),
            ColumnName.input.value: StringType(),
            ColumnName.output.value: StringType(),
            ColumnName.text.value: StringType(),
        },
        {
            ColumnName.instruction.value: StringType(),
            ColumnName.input.value: StringType(),
            ColumnName.output.value: StringType(),
        },
        {
            ColumnName.instruction.value: StringType(),
            ColumnName.output.value: StringType(),
        },
    ]
}

BuildLoraModelCallable = Callable[..., torch.nn.Module]
BuildTokenizerCallable = Callable[..., Llama3Tokenizer]


async def get_model_definition(
    model_id: str,
) -> BuildLoraModelCallable:
    model = resolve_model(model_id)
    if model is None or model.core_model_id.value not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_id} is not supported.")
    return MODEL_CONFIGS[model.core_model_id.value]["model_definition"]


async def get_tokenizer_type(
    model_id: str,
) -> BuildTokenizerCallable:
    model = resolve_model(model_id)
    return MODEL_CONFIGS[model.core_model_id.value]["tokenizer_type"]


async def get_checkpointer_model_type(
    model_id: str,
) -> str:
    """
    checkpointer model type is used in checkpointer for some special treatment on some specific model types
    For example, llama3.2 model tied weights (https://github.com/pytorch/torchtune/blob/main/torchtune/training/checkpointing/_checkpointer.py#L1041)
    """
    model = resolve_model(model_id)
    return MODEL_CONFIGS[model.core_model_id.value]["checkpoint_type"]


async def validate_input_dataset_schema(
    datasets_api: Datasets,
    dataset_id: str,
    dataset_type: str,
) -> None:
    dataset_def = await datasets_api.get_dataset(dataset_id=dataset_id)
    if not dataset_def.dataset_schema or len(dataset_def.dataset_schema) == 0:
        raise ValueError(f"Dataset {dataset_id} does not have a schema defined.")

    if dataset_def.dataset_schema not in EXPECTED_DATASET_SCHEMA[dataset_type]:
        raise ValueError(
            f"Dataset {dataset_id} does not have a correct input schema in {EXPECTED_DATASET_SCHEMA[dataset_type]}"
        )
