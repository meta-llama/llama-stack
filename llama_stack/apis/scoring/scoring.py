# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Protocol, runtime_checkable

from llama_models.schema_utils import json_schema_type, webmethod
from pydantic import BaseModel

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.scoring_functions import *  # noqa: F403


# mapping of metric to value
ScoringResultRow = Dict[str, Any]


@json_schema_type
class ScoringResult(BaseModel):
    score_rows: List[ScoringResultRow]
    # aggregated metrics to value
    aggregated_results: Dict[str, Any]


@json_schema_type
class ScoreBatchResponse(BaseModel):
    dataset_id: Optional[str] = None
    results: Dict[str, ScoringResult]


@json_schema_type
class ScoreResponse(BaseModel):
    # each key in the dict is a scoring function name
    results: Dict[str, ScoringResult]


class ScoringFunctionStore(Protocol):
    def get_scoring_function(self, name: str) -> ScoringFnDefWithProvider: ...


@runtime_checkable
class Scoring(Protocol):
    scoring_function_store: ScoringFunctionStore

    @webmethod(route="/scoring/score_batch")
    async def score_batch(
        self,
        dataset_id: str,
        scoring_functions: List[str],
        save_results_dataset: bool = False,
    ) -> ScoreBatchResponse: ...

    @webmethod(route="/scoring/score")
    async def score(
        self, input_rows: List[Dict[str, Any]], scoring_functions: List[str]
    ) -> ScoreResponse: ...
