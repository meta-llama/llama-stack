# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_models.llama3.api.datatypes import URL
from llama_stack.apis.common.type_system import ChatCompletionInputType, StringType
from llama_stack.apis.datasetio import DatasetDef

llamastack_mmlu = DatasetDef(
    identifier="llamastack_mmlu",
    url=URL(
        uri="https://huggingface.co/datasets/llamastack/Llama-3.2-1B-Instruct-evals"
    ),
    dataset_schema={
        "input_query": StringType(),
        "expected_answer": StringType(),
        "chat_completion_input": ChatCompletionInputType(),
    },
    metadata={
        "path": "llamastack/Llama-3.2-1B-Instruct-evals",
        "name": "Llama-3.2-1B-Instruct-evals__mmlu__details",
        "split": "train",
    },
)
