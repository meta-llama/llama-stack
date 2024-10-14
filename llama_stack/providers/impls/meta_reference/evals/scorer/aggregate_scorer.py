# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from llama_stack.apis.evals.evals import BaseScorer, EvalResult, SingleEvalResult
from llama_stack.apis.datasets.datasets import *  # noqa: F401 F403


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
