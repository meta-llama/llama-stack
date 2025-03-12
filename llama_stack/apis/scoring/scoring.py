# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any, Dict, List, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llama_stack.apis.common.job_types import CommonJobFields
from llama_stack.apis.scoring_functions import ScoringFn, ScoringFnParams
from llama_stack.schema_utils import json_schema_type, webmethod

# mapping of metric to value
ScoringResultRow = Dict[str, Any]


@json_schema_type
class ScoringResult(BaseModel):
    """
    A scoring result for a single row.

    :param score_rows: The scoring result for each row. Each row is a map of column name to value.
    :param aggregated_results: Map of metric name to aggregated value
    """

    score_rows: List[ScoringResultRow]
    # aggregated metrics to value
    aggregated_results: Dict[str, Any]


@json_schema_type
class ScoreBatchResponse(BaseModel):
    dataset_id: Optional[str] = None
    results: Dict[str, ScoringResult]


@json_schema_type
class ScoreResponse(BaseModel):
    """
    The response from scoring.

    :param results: A map of scoring function name to ScoringResult.
    """

    # each key in the dict is a scoring function name
    results: Dict[str, ScoringResult]


@json_schema_type
class ScoringJob(CommonJobFields):
    """The ScoringJob object representing a scoring job that was created through API."""

    type: Literal["scoring"] = "scoring"
    # TODO: result files or result datasets ids?
    result_files: List[str] = Field(
        default_factory=list,
        description="Result files of an scoring run. Which can be queried for results.",
    )


class ScoringFunctionStore(Protocol):
    def get_scoring_function(self, scoring_fn_id: str) -> ScoringFn: ...


@runtime_checkable
class Scoring(Protocol):
    scoring_function_store: ScoringFunctionStore

    @webmethod(route="/scoring/jobs", method="POST")
    async def score_dataset(
        self,
        dataset_id: str,
        scoring_functions: List[ScoringFnParams],
    ) -> ScoringJob: ...

    @webmethod(route="/scoring/rows", method="POST")
    async def score_rows(
        self,
        input_rows: List[Dict[str, Any]],
        scoring_functions: List[ScoringFnParams],
    ) -> ScoreResponse:
        """Score a list of rows.

        :param input_rows: The rows to score.
        :param scoring_functions: The scoring functions to use for the scoring.
        :return: ScoreResponse object containing rows and aggregated results
        """
        ...

    @webmethod(route="/scoring/jobs/{job_id}", method="GET")
    async def get_job(self, job_id: str) -> Optional[ScoringJob]:
        """Get the ScoringJob object for a given job id.

        :param job_id: The ID of the job to get the status of.
        :return: ScoringJob object indicating its status
        """
        ...

    @webmethod(route="/scoring/jobs/{job_id}", method="DELETE")
    async def cancel_job(self, job_id: str) -> None:
        """Cancel a job.

        :param job_id: The ID of the job to cancel.
        """
        ...
