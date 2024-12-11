# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Optional

from llama_stack.apis.scoring import ScoringResultRow

from llama_stack.apis.scoring_functions import ScoringFnParams
from llama_stack.providers.utils.scoring.aggregation_utils import aggregate_metrics
from llama_stack.providers.utils.scoring.base_scoring_fn import BaseScoringFn

from .fn_defs.equality import equality


class EqualityScoringFn(BaseScoringFn):
    """
    A scoring_fn that assigns a score of 1.0 if the input string matches the target string, and 0.0 otherwise.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supported_fn_defs_registry = {
            equality.identifier: equality,
        }

    async def score_row(
        self,
        input_row: Dict[str, Any],
        scoring_fn_identifier: Optional[str] = "equality",
        scoring_params: Optional[ScoringFnParams] = None,
    ) -> ScoringResultRow:
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

    async def aggregate(
        self,
        scoring_results: List[ScoringResultRow],
        scoring_fn_identifier: Optional[str] = None,
        scoring_params: Optional[ScoringFnParams] = None,
    ) -> Dict[str, Any]:
        params = self.supported_fn_defs_registry[scoring_fn_identifier].params
        if scoring_params is not None:
            params = scoring_params

        aggregation_functions = []
        if (
            params
            and hasattr(params, "aggregation_functions")
            and params.aggregation_functions
        ):
            aggregation_functions.extend(params.aggregation_functions)
        return aggregate_metrics(scoring_results, aggregation_functions)
