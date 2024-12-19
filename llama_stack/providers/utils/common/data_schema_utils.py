# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum

from llama_stack.apis.common.type_system import (
    ChatCompletionInputType,
    CompletionInputType,
    StringType,
)


class ColumnName(Enum):
    input_query = "input_query"
    expected_answer = "expected_answer"
    chat_completion_input = "chat_completion_input"
    completion_input = "completion_input"
    generated_answer = "generated_answer"
    context = "context"


def get_expected_schema_for_scoring():
    return [
        {
            ColumnName.input_query.value: StringType(),
            ColumnName.expected_answer.value: StringType(),
            ColumnName.generated_answer.value: StringType(),
        },
        {
            ColumnName.input_query.value: StringType(),
            ColumnName.expected_answer.value: StringType(),
            ColumnName.generated_answer.value: StringType(),
            ColumnName.context.value: StringType(),
        },
    ]


def get_expected_schema_for_eval():
    return [
        {
            ColumnName.input_query.value: StringType(),
            ColumnName.expected_answer.value: StringType(),
            ColumnName.chat_completion_input.value: ChatCompletionInputType(),
        },
        {
            ColumnName.input_query.value: StringType(),
            ColumnName.expected_answer.value: StringType(),
            ColumnName.completion_input.value: CompletionInputType(),
        },
    ]
