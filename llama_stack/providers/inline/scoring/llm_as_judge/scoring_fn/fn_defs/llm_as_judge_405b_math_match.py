# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.common.type_system import NumberType
from llama_stack.apis.scoring_functions import (
    AggregationFunctionType,
    LLMAsJudgeScoringFnParams,
    ScoringFn,
)

EQUALITY_TEMPLATE = r"""
Look at the following two expressions (answers to a math problem) and judge whether they are equivalent. Only perform trivial simplifications

Examples:

    Expression 1: $2x+3$
    Expression 2: $3+2x$

Yes

    Expression 1: 3/2
    Expression 2: 1.5

Yes

    Expression 1: $x^2+2x+1$
    Expression 2: $y^2+2y+1$

No

    Expression 1: $x^2+2x+1$
    Expression 2: $(x+1)^2$

Yes

    Expression 1: 3245/5
    Expression 2: 649

No
(these are actually equal, don't mark them equivalent if you need to do nontrivial simplifications)

    Expression 1: 2/(-3)
    Expression 2: -2/3

Yes
(trivial simplifications are allowed)

    Expression 1: 72 degrees
    Expression 2: 72

Yes
(give benefit of the doubt to units)

    Expression 1: 64
    Expression 2: 64 square feet

Yes
(give benefit of the doubt to units)

---

YOUR TASK


Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

    Expression 1: %(expression1)s
    Expression 2: %(expression2)s
""".strip()


llm_as_judge_405b_math_match = ScoringFn(
    identifier="llm-as-judge::405b-math-match",
    description="Llm As Judge Scoring Function for Math Related Benchmark https://github.com/openai/simple-evals/blob/main/math_eval.py)",
    return_type=NumberType(),
    provider_id="llm-as-judge",
    provider_resource_id="llm-as-judge-405b-math-match",
    params=LLMAsJudgeScoringFnParams(
        judge_model="meta-llama/Llama-3.1-405B-Instruct",
        prompt_template=EQUALITY_TEMPLATE,
        aggregation_functions=[AggregationFunctionType.accuracy],
    ),
)
