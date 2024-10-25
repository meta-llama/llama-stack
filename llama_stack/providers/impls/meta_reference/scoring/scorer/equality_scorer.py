# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.providers.impls.meta_reference.scoring.scorer.base_scorer import (
    BaseScorer,
)
from llama_stack.apis.scoring_functions import *  # noqa: F401, F403
from llama_stack.apis.scoring import *  # noqa: F401, F403
from llama_stack.apis.common.type_system import *  # noqa: F403


class EqualityScorer(BaseScorer):
    """
    A scorer that assigns a score of 1.0 if the input string matches the target string, and 0.0 otherwise.
    """

    scoring_function_def = ScoringFunctionDef(
        identifier="equality",
        description="Returns 1.0 if the input is equal to the target, 0.0 otherwise.",
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
        score = 1.0 if expected_answer == generated_answer else 0.0
        return {
            "score": score,
        }

    def aggregate(self, scoring_results: List[ScoringResultRow]) -> Dict[str, Any]:
        assert len(scoring_results) > 0, "Empty scoring results provided."
        num_correct = sum(result["score"] for result in scoring_results)
        avg_score = num_correct / len(scoring_results)

        return {
            "accuracy": avg_score,
            "num_correct": num_correct,
            "num_total": len(scoring_results),
        }
