# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import Any, Dict, List

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


class DataSchemaValidatorMixin:
    def validate_dataset_schema_for_scoring(self, dataset_schema: Dict[str, Any]):
        self.validate_dataset_schema(
            dataset_schema, self.get_expected_schema_for_scoring()
        )

    def validate_dataset_schema_for_eval(self, dataset_schema: Dict[str, Any]):
        self.validate_dataset_schema(
            dataset_schema, self.get_expected_schema_for_eval()
        )

    def validate_row_schema_for_scoring(self, input_row: Dict[str, Any]):
        self.validate_row_schema(input_row, self.get_expected_schema_for_scoring())

    def validate_row_schema_for_eval(self, input_row: Dict[str, Any]):
        self.validate_row_schema(input_row, self.get_expected_schema_for_eval())

    def get_expected_schema_for_scoring(self):
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

    def get_expected_schema_for_eval(self):
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

    def validate_dataset_schema(
        self,
        dataset_schema: Dict[str, Any],
        expected_schemas: List[Dict[str, Any]],
    ):
        if dataset_schema not in expected_schemas:
            raise ValueError(
                f"Dataset does not have a correct input schema in {expected_schemas}"
            )

    def validate_row_schema(
        self,
        input_row: Dict[str, Any],
        expected_schemas: List[Dict[str, Any]],
    ):
        for schema in expected_schemas:
            if all(key in input_row for key in schema):
                return

        raise ValueError(
            f"Input row {input_row} does not match any of the expected schemas in {expected_schemas}"
        )
