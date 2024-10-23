# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from llama_stack.apis.scoring_functions import *  # noqa: F401, F403
from llama_stack.apis.scoring import *  # noqa: F401, F403


class BaseScorer(ABC):
    """
    Base interface class for all meta-reference scorers.
    Each scorer needs to implement the following methods:
    - score_row(self, row)
    - aggregate(self, scorer_results)
    """

    scoring_function_def: DeterministicFunctionDef

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def __str__(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def score_row(self, input_row: Dict[str, Any]) -> ScoringResult:
        raise NotImplementedError()

    @abstractmethod
    def aggregate(self, scoring_results: List[ScoringResult]) -> ScoringResult:
        raise NotImplementedError()

    def score(self, input_rows: List[Dict[str, Any]]) -> List[ScoringResult]:
        return [self.score_row(input_row) for input_row in input_rows]
