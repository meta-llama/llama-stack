# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from llama_stack.apis.scoring_functions import *  # noqa: F401, F403
from llama_stack.apis.scoring import *  # noqa: F401, F403


class BaseScoringFn(ABC):
    """
    Base interface class for all meta-reference scoring_fns.
    Each scoring_fn needs to implement the following methods:
    - score_row(self, row)
    - aggregate(self, scoring_fn_results)
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.supported_fn_defs_registry = {}

    def __str__(self) -> str:
        return self.__class__.__name__

    def get_supported_scoring_fn_defs(self) -> List[ScoringFnDef]:
        return [x for x in self.supported_fn_defs_registry.values()]

    def register_scoring_fn_def(self, scoring_fn_def: ScoringFnDef) -> None:
        if scoring_fn_def.identifier in self.supported_fn_defs_registry:
            raise ValueError(
                f"Scoring function def with identifier {scoring_fn_def.identifier} already exists."
            )
        self.supported_fn_defs_registry[scoring_fn_def.identifier] = scoring_fn_def

    @abstractmethod
    async def score_row(
        self, input_row: Dict[str, Any], scoring_fn_identifier: Optional[str] = None
    ) -> ScoringResultRow:
        raise NotImplementedError()

    @abstractmethod
    async def aggregate(
        self, scoring_results: List[ScoringResultRow]
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    async def score(
        self,
        input_rows: List[Dict[str, Any]],
        scoring_fn_identifier: Optional[str] = None,
    ) -> List[ScoringResultRow]:
        return [
            await self.score_row(input_row, scoring_fn_identifier)
            for input_row in input_rows
        ]
