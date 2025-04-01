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

from typing import Any

from llama_stack.apis.common.type_system import (
    ChatCompletionInputType,
    DialogType,
    StringType,
)
from llama_stack.apis.datasets import Datasets
from llama_stack.providers.utils.common.data_schema_validator import (
    ColumnName,
    validate_dataset_schema,
)

EXPECTED_DATASET_SCHEMA: dict[str, list[dict[str, Any]]] = {
    "instruct": [
        {
            ColumnName.chat_completion_input.value: ChatCompletionInputType(),
            ColumnName.expected_answer.value: StringType(),
        }
    ],
    "dialog": [
        {
            ColumnName.dialog.value: DialogType(),
        }
    ],
}


async def validate_input_dataset_schema(
    datasets_api: Datasets,
    dataset_id: str,
    dataset_type: str,
) -> None:
    dataset_def = await datasets_api.get_dataset(dataset_id=dataset_id)
    if not dataset_def:
        raise ValueError(f"Dataset {dataset_id} does not exist.")

    if not dataset_def.dataset_schema or len(dataset_def.dataset_schema) == 0:
        raise ValueError(f"Dataset {dataset_id} does not have a schema defined.")

    if dataset_type not in EXPECTED_DATASET_SCHEMA:
        raise ValueError(f"Dataset type {dataset_type} is not supported.")

    validate_dataset_schema(dataset_def.dataset_schema, EXPECTED_DATASET_SCHEMA[dataset_type])
