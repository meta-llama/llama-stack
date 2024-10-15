# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import numpy as np

from llama_stack.apis.evals.evals import BaseScorer, EvalResult, SingleEvalResult
from llama_stack.apis.datasets.datasets import *  # noqa: F401 F403
from autoevals.llm import *  # noqa: F403
from autoevals.ragas import *  # noqa: F403


class BraintrustFactualityScorer(BaseScorer[ScorerInputSample]):
    def score_sample(self, scorer_input_sample: ScorerInputSample) -> SingleEvalResult:
        input_query = scorer_input_sample.input_query
        extracted_answer = scorer_input_sample.generated_answer
        expected_answer = scorer_input_sample.expected_answer

        evaluator = Factuality()
        result = evaluator(output, expected, input=input_query)
        factuality = result.score
        return SingleEvalResult(score_data={"factuality": factuality})

    def aggregate_results(self, eval_results: List[SingleEvalResult]) -> EvalResult:
        avg_score = np.average(
            [result.score_data["factuality"] for result in eval_results]
        )

        return EvalResult(
            metrics={
                "avg_factuality_score": avg_score,
            }
        )


class BraintrustAnswerCorrectnessScorer(BaseScorer[ScorerInputSample]):
    def score_sample(self, scorer_input_sample: ScorerInputSample) -> SingleEvalResult:
        input_query = scorer_input_sample.input_query
        extracted_answer = scorer_input_sample.generated_answer
        expected_answer = scorer_input_sample.expected_answer

        evaluator = AnswerCorrectness()
        result = evaluator(output, expected, input=input_query)
        correctness = result.score
        return SingleEvalResult(score_data={"answer_correctness": correctness})

    def aggregate_results(self, eval_results: List[SingleEvalResult]) -> EvalResult:
        avg_score = np.average(
            [result.score_data["answer_correctness"] for result in eval_results]
        )

        return EvalResult(
            metrics={
                "avg_correctness_score": avg_score,
            }
        )
