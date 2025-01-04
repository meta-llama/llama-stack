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
from llama_models.datatypes import Model
from llama_models.sku_list import resolve_model
from llama_stack.apis.common.type_system import ParamType, StringType
from llama_stack.apis.datasets import Datasets

from pydantic import BaseModel

from torchtune.models.llama3 import llama3_tokenizer
from torchtune.models.llama3._tokenizer import Llama3Tokenizer
from torchtune.models.llama3_1 import lora_llama3_1_8b
from torchtune.models.llama3_2 import lora_llama3_2_3b


class ColumnName(Enum):
    instruction = "instruction"
    input = "input"
    output = "output"
    text = "text"


class ModelConfig(BaseModel):
    model_definition: Any
    tokenizer_type: Any
    checkpoint_type: str


class DatasetSchema(BaseModel):
    alpaca: List[Dict[str, ParamType]]


MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "Llama3.2-3B-Instruct": ModelConfig(
        model_definition=lora_llama3_2_3b,
        tokenizer_type=llama3_tokenizer,
        checkpoint_type="LLAMA3_2",
    ),
    "Llama3.1-8B-Instruct": ModelConfig(
        model_definition=lora_llama3_1_8b,
        tokenizer_type=llama3_tokenizer,
        checkpoint_type="LLAMA3",
    ),
}


EXPECTED_DATASET_SCHEMA = DatasetSchema(
    alpaca=[
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
)

BuildLoraModelCallable = Callable[..., torch.nn.Module]
BuildTokenizerCallable = Callable[..., Llama3Tokenizer]


def _validate_model_id(model_id: str) -> Model:
    model = resolve_model(model_id)
    if model is None or model.core_model_id.value not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_id} is not supported.")
    return model


async def get_model_definition(
    model_id: str,
) -> BuildLoraModelCallable:
    model = _validate_model_id(model_id)
    model_config = MODEL_CONFIGS[model.core_model_id.value]
    if not hasattr(model_config, "model_definition"):
        raise ValueError(f"Model {model_id} does not have model definition.")
    return model_config.model_definition


async def get_tokenizer_type(
    model_id: str,
) -> BuildTokenizerCallable:
    model = _validate_model_id(model_id)
    model_config = MODEL_CONFIGS[model.core_model_id.value]
    if not hasattr(model_config, "tokenizer_type"):
        raise ValueError(f"Model {model_id} does not have tokenizer_type.")
    return model_config.tokenizer_type


async def get_checkpointer_model_type(
    model_id: str,
) -> str:
    """
    checkpointer model type is used in checkpointer for some special treatment on some specific model types
    For example, llama3.2 model tied weights (https://github.com/pytorch/torchtune/blob/main/torchtune/training/checkpointing/_checkpointer.py#L1041)
    """
    model = _validate_model_id(model_id)
    model_config = MODEL_CONFIGS[model.core_model_id.value]
    if not hasattr(model_config, "checkpoint_type"):
        raise ValueError(f"Model {model_id} does not have checkpoint_type.")
    return model_config.checkpoint_type


async def validate_input_dataset_schema(
    datasets_api: Datasets,
    dataset_id: str,
    dataset_type: str,
) -> None:
    dataset_def = await datasets_api.get_dataset(dataset_id=dataset_id)
    if not dataset_def.dataset_schema or len(dataset_def.dataset_schema) == 0:
        raise ValueError(f"Dataset {dataset_id} does not have a schema defined.")

    if not hasattr(EXPECTED_DATASET_SCHEMA, dataset_type):
        raise ValueError(f"Dataset type {dataset_type} is not supported.")

    if dataset_def.dataset_schema not in getattr(EXPECTED_DATASET_SCHEMA, dataset_type):
        raise ValueError(
            f"Dataset {dataset_id} does not have a correct input schema in {getattr(EXPECTED_DATASET_SCHEMA, dataset_type)}"
        )
