# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pathlib import Path

from typing import Any, Dict, List, Optional

# TODO: move the common base out from meta-reference into common
from llama_stack.providers.impls.meta_reference.scoring.scoring_fn.base_scoring_fn import (
    BaseScoringFn,
)
from llama_stack.providers.impls.meta_reference.scoring.scoring_fn.common import (
    aggregate_average,
)
from llama_stack.apis.scoring import *  # noqa: F403
from llama_stack.apis.scoring_functions import *  # noqa: F403
from llama_stack.apis.common.type_system import *  # noqa: F403
from autoevals.llm import Factuality
from autoevals.ragas import AnswerCorrectness


BRAINTRUST_FN_DEFS_PATH = Path(__file__).parent / "fn_defs"


class BraintrustScoringFn(BaseScoringFn):
    """
    Test whether an output is factual, compared to an original (`expected`) value.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.braintrust_evaluators = {
            "braintrust::factuality": Factuality(),
            "braintrust::answer-correctness": AnswerCorrectness(),
        }
        self.defs_paths = [
            str(x) for x in sorted(BRAINTRUST_FN_DEFS_PATH.glob("*.json"))
        ]

    async def score_row(
        self,
        input_row: Dict[str, Any],
        scoring_fn_identifier: Optional[str] = None,
    ) -> ScoringResultRow:
        assert scoring_fn_identifier is not None, "scoring_fn_identifier cannot be None"
        expected_answer = input_row["expected_answer"]
        generated_answer = input_row["generated_answer"]
        input_query = input_row["input_query"]
        evaluator = self.braintrust_evaluators[scoring_fn_identifier]

        result = evaluator(generated_answer, expected_answer, input=input_query)
        score = result.score
        return {"score": score, "metadata": result.metadata}

    async def aggregate(
        self, scoring_results: List[ScoringResultRow]
    ) -> Dict[str, Any]:
        return aggregate_average(scoring_results)
