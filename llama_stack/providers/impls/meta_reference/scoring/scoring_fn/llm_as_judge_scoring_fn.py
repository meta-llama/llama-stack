# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from openai import OpenAI

from llama_stack.apis.inference.inference import Inference

from .base_scoring_fn import BaseScoringFn
from llama_stack.apis.scoring_functions import *  # noqa: F401, F403
from llama_stack.apis.scoring import *  # noqa: F401, F403
from llama_stack.apis.common.type_system import *  # noqa: F403
import re

from .common import aggregate_average, aggregate_count
from .fn_defs.llm_as_judge_8b_correctness import llm_as_judge_8b_correctness
from .fn_defs.llm_as_judge_openai_simpleqa import llm_as_judge_openai_simpleqa


class LlmAsJudgeScoringFn(BaseScoringFn):
    """
    A scoring_fn using LLM as Judge to produce score
    """

    def __init__(self, inference_api: Inference, *arg, **kwargs) -> None:
        super().__init__(*arg, **kwargs)
        self.inference_api = inference_api
        self.supported_fn_defs_registry = {
            llm_as_judge_8b_correctness.identifier: llm_as_judge_8b_correctness,
            llm_as_judge_openai_simpleqa.identifier: llm_as_judge_openai_simpleqa,
        }

    async def score_row(
        self,
        input_row: Dict[str, Any],
        scoring_fn_identifier: Optional[str] = None,
    ) -> ScoringResultRow:
        assert (
            scoring_fn_identifier is not None
        ), "Scoring function identifier not found."
        fn_def = self.supported_fn_defs_registry[scoring_fn_identifier]
        assert fn_def.context is not None, f"LLMAsJudgeContext not found for {fn_def}."
        assert (
            fn_def.context.prompt_template is not None
        ), "LLM Judge prompt_template not found."
        assert (
            fn_def.context.judge_score_regex is not None
        ), "LLM Judge judge_score_regex not found."

        input_query = input_row["input_query"]
        expected_answer = input_row["expected_answer"]
        generated_answer = input_row["generated_answer"]

        judge_input_msg = fn_def.context.prompt_template.format(
            input_query=input_query,
            expected_answer=expected_answer,
            generated_answer=generated_answer,
        )

        # TODO: we need a registry of 3P models as judge
        if fn_def.context.judge_model in ["gpt-4o"]:
            openai_client = OpenAI()
            judge_response = openai_client.chat.completions.create(
                model=fn_def.context.judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": judge_input_msg,
                    },
                ],
            )
            content = judge_response.choices[0].message.content
        else:
            judge_response = await self.inference_api.chat_completion(
                model=fn_def.context.judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": judge_input_msg,
                    }
                ],
            )
            content = judge_response.completion_message.content

        rating_regexs = fn_def.context.judge_score_regex
        judge_grade_metrics = fn_def.context.judge_grade_metrics

        judge_score = None
        judge_grade = None
        result = {"judge_feedback": content}

        for regex in rating_regexs:
            match = re.search(regex, content)
            if match:
                if judge_grade_metrics:
                    judge_score = match.group(1)
                    judge_grade = judge_grade_metrics.get(judge_score, None)
                else:
                    judge_score = int(match.group(1))
                break

        result["score"] = judge_score
        result["grade"] = judge_grade

        return result

    async def aggregate(
        self,
        scoring_results: List[ScoringResultRow],
        scoring_fn_identifier: Optional[str] = None,
    ) -> Dict[str, Any]:
        fn_def = self.supported_fn_defs_registry[scoring_fn_identifier]
        judge_grade_metrics = fn_def.context.judge_grade_metrics
        if judge_grade_metrics:
            return aggregate_count(scoring_results)
        else:
            return aggregate_average(scoring_results)
