# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack.apis.scoring_functions import *  # noqa: F401, F403
from llama_stack.apis.scoring import *  # noqa: F401, F403
from llama_stack.apis.common.type_system import NumberType

JUDGE_PROMPT = """
You will be given a question, a expected_answer, and a system_answer.
Your task is to provide a 'total rating' scoring how well the system_answer answers compared with ground truth in expected_answer in terms of factual correctness to the question.
Give your answer as a integer on a scale of 0 to 5, where 0 means that the system_answer is not correct at all compared with expected_answer, and 5 means that the answer completely and correctly answers the question.
Provide your feedback as follows:
Feedback:::
Total rating: (your rating, as a int between 0 and 5)
Now here are the question, expected_answer, system_answer.
Question: {input_query}
Expected Answer: {expected_answer}
System Answer: {generated_answer}
Feedback:::
Total rating:
"""

llm_as_judge_8b_correctness = ScoringFnDef(
    identifier="meta-reference::llm_as_judge_8b_correctness",
    description="Llm As Judge Scoring Function",
    parameters=[],
    return_type=NumberType(),
    context=LLMAsJudgeContext(
        prompt_template=JUDGE_PROMPT,
        judge_model="Llama3.1-8B-Instruct",
        judge_score_regex=[r"Total rating: (\d+)", r"rating: (\d+)", r"Rating: (\d+)"],
    ),
)
