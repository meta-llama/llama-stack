# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import AbstractSet, Dict

from llama_stack.apis.evals import BaseScorer


class ScorerRegistry:
    _REGISTRY: Dict[str, BaseScorer] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return ScorerRegistry._REGISTRY.keys()

    @staticmethod
    def register(name: str, scorer: BaseScorer) -> None:
        if name in ScorerRegistry._REGISTRY:
            raise ValueError(f"Task {name} already exists.")
        ScorerRegistry._REGISTRY[name] = task

    @staticmethod
    def get_scorer(name: str) -> BaseScorer:
        if name not in ScorerRegistry._REGISTRY:
            raise ValueError(f"Task {name} not found.")
        return ScorerRegistry._REGISTRY[name]

    @staticmethod
    def reset() -> None:
        ScorerRegistry._REGISTRY = {}
