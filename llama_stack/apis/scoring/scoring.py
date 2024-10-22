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


ScoringResult = Dict[str, Any]


@json_schema_type
class ScoreBatchResponse(BaseModel):
    dataset_id: str


@json_schema_type
class ScoreResponse(BaseModel):
    # each key in the dict is a scoring function name
    results: List[Dict[str, ScoringResult]]


class ScoringFunctionStore(Protocol):
    def get_scoring_function(self, name: str) -> ScoringFunctionDefWithProvider: ...


@runtime_checkable
class Scoring(Protocol):
    scoring_function_store: ScoringFunctionStore

    @webmethod(route="/scoring/score_batch")
    async def score_batch(
        self, dataset_id: str, scoring_functions: List[str]
    ) -> ScoreBatchResponse: ...

    @webmethod(route="/scoring/score")
    async def score(
        self, input_rows: List[Dict[str, Any]], scoring_functions: List[str]
    ) -> ScoreResponse: ...
