# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import random

from llama_stack.apis.evals.evals import BaseScorer, EvalResult, SingleEvalResult
from llama_stack.apis.dataset.dataset import *  # noqa: F401 F403


class AggregateScorer(BaseScorer[ScorerInputSample]):
    def __init__(self, scorers: List[BaseScorer[ScorerInputSample]]):
        self.scorers = scorers

    def score_sample(self, scorer_input_sample: ScorerInputSample) -> SingleEvalResult:
        all_score_data = {}
        for scorer in self.scorers:
            score_data = scorer.score_sample(scorer_input_sample).score_data
            for k, v in score_data.items():
                all_score_data[k] = v

        return SingleEvalResult(
            score_data=all_score_data,
        )

    def aggregate_results(self, eval_results: List[SingleEvalResult]) -> EvalResult:
        all_metrics = {}

        for scorer in self.scorers:
            metrics = scorer.aggregate_results(eval_results).metrics
            for k, v in metrics.items():
                all_metrics[f"{scorer.__class__.__name__}:{k}"] = v

        return EvalResult(
            metrics=all_metrics,
        )


class RandomScorer(BaseScorer[ScorerInputSample]):
    def score_sample(self, scorer_input_sample: ScorerInputSample) -> SingleEvalResult:
        return SingleEvalResult(score_data={"random": random.random()})

    def aggregate_results(self, eval_results: List[SingleEvalResult]) -> EvalResult:
        avg_random = sum(
            [result.score_data["random"] for result in eval_results]
        ) / len(eval_results)
        max_random = max([result.score_data["random"] for result in eval_results])
        return EvalResult(
            metrics={
                "avg_random": avg_random,
                "max_random": max_random,
            }
        )


class AccuracyScorer(BaseScorer[ScorerInputSample]):
    def score_sample(self, scorer_input_sample: ScorerInputSample) -> SingleEvalResult:
        extracted_answer = scorer_input_sample.generation_output.transformed_generation
        expected_answer = scorer_input_sample.expected_output

        accuracy = (
            1.0 if extracted_answer and extracted_answer == expected_answer else 0.0
        )

        return SingleEvalResult(score_data={"accuracy": accuracy})

    def aggregate_results(self, eval_results: List[SingleEvalResult]) -> EvalResult:
        num_correct = sum([result.score_data["accuracy"] for result in eval_results])
        num_total = len(eval_results)

        return EvalResult(
            metrics={
                "avg_accuracy": num_correct / num_total,
                "num_correct": num_correct,
                "num_total": num_total,
            }
        )
