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

JUDGE_PROMPT = """
You will be given a question, a expected_answer, and a system_answer.
Your task is to provide a 'total rating' scoring how well the system_answer answers compared with ground truth in expected_answer in terms of factual correctness to the question.
Give your answer as a integer on a scale of 0 to 5, where 0 means that the system_answer is not correct at all compared with expected_answer, and 5 means that the answer completely and correctly answers the question.
Provide your feedback as follows:
Feedback:::
Total rating: (your rating, as a int between 0 and 5)
Now here are the question, expected_answer, system_answer.
Question: {question}
Expected Answer: {expected_answer}
System Answer: {answer}
Feedback:::
Total rating:
"""


class LlmAsJudgeScoringFn(BaseScoringFn):
    """
    A scoring_fn that assigns
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scoring_fn_def_registry = {}

    def register_scoring_def(self, scoring_fn_def: ScoringFnDef) -> None:
        self.scoring_function_def_registry[scoring_fn_def.identifier] = scoring_fn_def

    async def score_row(self, input_row: Dict[str, Any]) -> ScoringResultRow:
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
        self, scoring_results: List[ScoringResultRow]
    ) -> Dict[str, Any]:
        return aggregate_accuracy(scoring_results)
