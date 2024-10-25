# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.impls.meta_reference.scoring.scoring_fn.base_scoring_fn import (
    BaseScoringFn,
)
from llama_stack.apis.scoring_functions import *  # noqa: F401, F403
from llama_stack.apis.scoring import *  # noqa: F401, F403
from llama_stack.apis.common.type_system import *  # noqa: F403
from llama_stack.providers.impls.meta_reference.scoring.scoring_fn.common import (
    aggregate_accuracy,
)


class SubsetOfScoringFn(BaseScoringFn):
    """
    A scoring_fn that assigns a score of 1.0 if the expected string is included in the generated string, and 0.0 otherwise.
    """

    scoring_function_def = ScoringFnDef(
        identifier="subset_of",
        description="Returns 1.0 if the expected is included in generated, 0.0 otherwise.",
        parameters=[],
        return_type=NumberType(),
    )

    def score_row(self, input_row: Dict[str, Any]) -> ScoringResultRow:
        assert "expected_answer" in input_row, "Expected answer not found in input row."
        assert (
            "generated_answer" in input_row
        ), "Generated answer not found in input row."

        expected_answer = input_row["expected_answer"]
        generated_answer = input_row["generated_answer"]
        score = 1.0 if expected_answer in generated_answer else 0.0
        return {
            "score": score,
        }

    def aggregate(self, scoring_results: List[ScoringResultRow]) -> Dict[str, Any]:
        return aggregate_accuracy(scoring_results)
