# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.llama3.api.datatypes import URL
from llama_stack.apis.common.type_system import StringType
from llama_stack.apis.datasetio import DatasetDef


llamastack_mmlu = DatasetDef(
    identifier="llamastack_mmlu",
    url=URL(uri="https://huggingface.co/datasets/yanxi0830/ls-mmlu"),
    dataset_schema={
        "expected_answer": StringType(),
        "input_query": StringType(),
        "generated_answer": StringType(),
    },
    metadata={"path": "yanxi0830/ls-mmlu", "split": "train"},
)
