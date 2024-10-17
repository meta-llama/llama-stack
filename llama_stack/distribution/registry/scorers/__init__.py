# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# TODO: make these import config based
from llama_stack.apis.evals import *  # noqa: F403
from llama_stack.providers.impls.meta_reference.evals.scorer.basic_scorers import *  # noqa: F403
from llama_stack.providers.impls.meta_reference.evals.scorer.braintrust_scorer import *  # noqa: F403
from llama_stack.providers.impls.meta_reference.evals.scorer.llm_judge_scorer import *  # noqa: F403

from ..registry import Registry

# TODO: make these import config based
ScorerRegistry = Registry[BaseScorer]()

SCORER_REGISTRY = {
    "accuracy": AccuracyScorer,
    "random": RandomScorer,
    "llamastack-llm-judge": LlamaStackLLMJudgeScorer,
    "braintrust::factuality": BraintrustFactualityScorer,
    "braintrust::answer-correctness": BraintrustAnswerCorrectnessScorer,
}

for k, v in SCORER_REGISTRY.items():
    ScorerRegistry.register(k, v)
